use std::{env, sync::Arc};

use qdrant_client::{
    Payload, Qdrant,
    qdrant::{
        CreateCollectionBuilder, PointStruct, QueryPointsBuilder, UpsertPointsBuilder,
        VectorParamsBuilder,
    },
};
use rig::{
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, EmbeddingModel as Model, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndexDyn,
};
use rig_qdrant::QdrantVectorStore;

use crate::topic::Utterance;

pub struct SemanticRouter {
    model: Model,
    qdrant: Arc<Qdrant>,
    vector_store: QdrantVectorStore<Model>,
}

const COLLECTION_NAME: &str = "SEMANTIC_ROUTING";
const COLLECTION_SIZE: usize = 1536;

impl SemanticRouter {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize OpenAI client.
        let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
        let openai_client = Client::new(&openai_api_key);

        let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

        // note that this assumes you're hosting Qdrant locally on localhost:6334
        let client = Qdrant::from_url("http://localhost:6334").build()?;
        let qdrant = Arc::new(client);

        // Create a collection with 1536 dimensions of it doesn't exist
        // Note: Make sure the dimensions match the size of the embedings model you are using
        if !qdrant.collection_exists(COLLECTION_NAME).await? {
            qdrant
                .create_collection(
                    CreateCollectionBuilder::new(COLLECTION_NAME).vectors_config(
                        VectorParamsBuilder::new(
                            COLLECTION_SIZE as u64,
                            qdrant_client::qdrant::Distance::Cosine,
                        ),
                    ),
                )
                .await?;
        }

        // 创建一个新的 Qdrant 客户端实例用于 vector_store
        let qdrant_for_store = Qdrant::from_url("http://localhost:6334").build()?;
        let query_params = QueryPointsBuilder::new(COLLECTION_NAME).with_payload(true);
        let vector_store =
            QdrantVectorStore::new(qdrant_for_store, model.clone(), query_params.build());

        Ok(Self {
            model,
            qdrant,
            vector_store,
        })
    }

    pub async fn embed_utterances(
        &self,
        utterances: Vec<Utterance>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let documents = EmbeddingsBuilder::new(self.model.clone())
            .documents(utterances)?
            .build()
            .await?;
        let points: Vec<PointStruct> = documents
            .into_iter()
            .map(|(d, embeddings)| {
                let vec: Vec<f32> = embeddings.first().vec.iter().map(|&x| x as f32).collect();
                PointStruct::new(
                    d.id.clone(),
                    vec,
                    Payload::try_from(serde_json::to_value(&d).unwrap()).unwrap(),
                )
            })
            .collect();

        self.qdrant
            .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, points))
            .await?;
        Ok(())
    }

    pub async fn query(&self, query: &str) -> Result<Utterance, Box<dyn std::error::Error>> {
        let results = self.vector_store.top_n(query, 1).await?;

        if results[0].0 <= 0.85 {
            return Err("No relevant snippet found.".into());
        }
        let utterance: Utterance = serde_json::from_value(results[0].2.clone())?;
        Ok(utterance)
    }
}
