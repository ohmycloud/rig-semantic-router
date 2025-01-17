use rig_semantic_router::{
    router::SemanticRouter,
    topic::{Topic, Utterance},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let router = SemanticRouter::new().await.unwrap();

    // create a vector of strings then iterate through the vector
    // and map them all to `Utterance` instance
    let bee_facts = vec![
        "Bees communicate with their hive mates through intricate dances that convey the location of nectar-rich flowers.",
            "A single bee can visit up to 5,000 flowers in a day, tirelessly collecting nectar and pollen.",
            "The queen bee can lay up to 2,000 eggs in a single day during peak season.",
    ].into_iter().map(|x| Topic::new("bees").new_utterance(x)).collect::<Vec<Utterance>>();

    // embed utterances into Qdrant
    router.embed_utterances(bee_facts).await.unwrap();

    let bee_answer = router
        .query("how many flowers does a bee visit in a day?")
        .await?;
    println!(
        "Topic: {}, content: {}",
        bee_answer.topic, bee_answer.content
    );

    // note that this query *should* error out as it's unrelated
    // in which case, we simply tell the user we can't help them
    match router.query("what is skibidi toilet").await {
        Ok(res) => println!("Unexpectedly found a topic: {}", res.topic),
        Err(_) => println!("Sorry, I can't help you with that."),
    };

    Ok(())
}

// https://dev.to/josh_mo_91f294fcef0333006/semantic-routing-with-qdrant-rig-rust-mj4
