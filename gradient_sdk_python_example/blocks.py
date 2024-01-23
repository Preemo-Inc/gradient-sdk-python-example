from typing import List, Mapping

from dotenv import load_dotenv

load_dotenv()
from gradientai import (
    AnalyzeSentimentParamsExample,
    ExtractParamsSchemaValue,
    ExtractParamsSchemaValueType,
    Gradient,
    Sentiment,
    SummarizeParamsExample,
    SummarizeParamsLength,
)


def run_answer_example(*, gradient: Gradient) -> None:
    document = (
        "When Apple released the Apple Watch in 2015, it was business "
        + "as usual for a company whose iPhone updates had become cultural "
        + "touchstones. Before the watch went on sale, Apple gave early "
        + "versions of it to celebrities like Beyoncé, featured it in fashion "
        + "publications like Vogue and streamed a splashy event on the "
        + "internet trumpeting its features."
    )
    question = "How was the Apple watch marketed?"

    print("==== Q & A ====")
    print(f"Document: {document}\n")
    print(f"Question: {question}\n")

    print("Answering question...")
    result = gradient.answer(
        question=question,
        source={
            "type": "document",
            "value": document,
        },
    )

    print(f"Answer: {result['answer']}")
    print("================\n")


def run_summarize_example(*, gradient: Gradient) -> None:
    document = (
        "In the days ahead of the Vision Pro's launch, Apple has heavily "
        + "promoted some of the apps destined for its spatial computing "
        + "headset. Download Disney Plus and watch movies from Tatooine! "
        + "Slack and Fantastical and Microsoft Office on your face! FaceTime "
        + "with your friends as a floating hologram! But it's increasingly "
        + "clear that the early success of the Vision Pro, and much of the "
        + "answer to the question of what this headset is actually for, will "
        + "come from a single app: Safari.\n\nThat's right, friends. Web "
        + "browsers are back. And Apple needs them more than ever if it wants "
        + "this $3,500 face computer to be a hit. Embracing the web will mean "
        + "threatening the very things that have made Apple so powerful and "
        + "so rich in the mobile era, but at least at first, the open web is "
        + "Apple's best chance to make its headset a winner. Because at least "
        + "so far, it seems developers are not exactly jumping to build new "
        + "apps for Apple's new platform."
    )
    examples: List[SummarizeParamsExample] = [
        {
            "document": (
                "Historically, Apple is unmatched in its ability to get app "
                + "makers to keep up with its newest stuff. When it releases "
                + "features for iPhones and iPads, a huge chunk of the App "
                + "Store supports those features within a few weeks. But so "
                + "far, developers appear to be taking their Vision Pro "
                + "development slowly. Exactly why varies across the App "
                + "Store, but there are a bunch of good reasons to choose "
                + "from. One is just that it's a new platform with new UI "
                + "ideas and usability concerns on a really expensive device "
                + "few people will have access to for a while. Sure, you can "
                + "more or less tick a box and port your iPad app to the "
                + "Vision Pro, but that may not be up to everyone's standards."
            ),
            "summary": (
                "Apple typically releases hardware first with app support "
                + "added over a few weeks. However, fewer developers are "
                + "supporting the Vision Pro over the first few weeks of "
                + "its release."
            ),
        },
    ]

    print("==== Document Summary (with examples) ====")
    print(f"Document: {document}\n")
    print(f"Examples: {examples}\n")

    print("Summarizing document...")
    result = gradient.summarize(
        document=document,
        examples=examples,
    )
    print(f"Summary: {result['summary']}")
    print("================\n")

    length = SummarizeParamsLength.MEDIUM

    print("==== Document Summary (with length) ====")
    print(f"Document: {document}\n")
    print(f"Length: {length}\n")

    print("Summarizing document...")
    result = gradient.summarize(document=document, length=length)
    print(f"Summary: {result['summary']}")
    print("================\n")


def run_analyze_sentiment_example(*, gradient: Gradient) -> None:
    document = (
        "Spotify has been railing against Apple's 30 percent cut of in-app "
        + "purchases for years."
    )
    examples: List[AnalyzeSentimentParamsExample] = [
        {
            "sentiment": Sentiment.NEGATIVE,
            "document": (
                "Netflix got a sweetheart deal from Apple years ago to share "
                + "only 15 percent of revenue but has recently been refusing "
                + "to participate in the Apple TV app's discovery feature and "
                + "has long since stopped allowing you to subscribe to "
                + "Netflix from your iOS device. "
            ),
        },
        {
            "sentiment": Sentiment.POSITIVE,
            "document": (
                "Over the last decade or so, we've all stopped opening "
                + "websites and started tapping app icons, but the age of "
                + "the URL might be coming back."
            )
        },
    ]

    print("==== Sentiment Analysis ====")
    print(f"Document: {document}\n")
    print(f"Examples: {examples}\n")

    print("Analyzing sentiment...")
    result = gradient.analyze_sentiment(document=document, examples=examples)
    print(f"Sentiment: {result['sentiment']}")
    print("================\n")


def run_personalize_example(*, gradient: Gradient) -> None:
    document = (
        "Harry Potter fans have been eagerly anticipating Hogwarts Legacy "
        + "since the game was first revealed in 2020, and unfortunately, it's "
        + "been a long wait to play the game. Hogwarts Legacy was originally "
        + "supposed to launch in 2021, but then its release date was pushed "
        + "back to 2022. For months, fans have been anticipating a holiday "
        + "2022 release date for the game, but now Hogwarts Legacy has been "
        + "delayed yet again, pushed back to 2023. With Hogwarts Legacy "
        + "getting its own State of Play presentation earlier this year, it "
        + "seemed like the game was on track to meet its planned 2022 release "
        + "date. After all, the Hogwarts Legacy gameplay footage shown during "
        + "the State of Play looked quite impressive, indicating that the "
        + "game's development was going well and nearing its end point."
    )
    audience_description = "Someone who loves playing action-adventure RPGs."

    print("==== Personalization ====")
    print(f"Document: {document}\n")
    print(f"Audience Description: {audience_description}\n")

    print("Personalizing document...")
    result = gradient.personalize(
        document=document, audience_description=audience_description
    )
    print(f"Personalized document: {result['personalized_document']}")
    print("================\n")


def run_extract_example(*, gradient: Gradient) -> None:
    document = (
        "When Apple released the Apple Watch in 2015, it was business as "
        + "usual for a company whose iPhone updates had become cultural "
        + "touchstones. Before the watch went on sale, Apple gave early "
        + "versions of it to celebrities like Beyoncé, featured it in fashion "
        + "publications like Vogue and streamed a splashy event on the "
        + "internet trumpeting its features."
    )
    schema_: Mapping[str, ExtractParamsSchemaValue] = {
        "company": {
            "type": ExtractParamsSchemaValueType.STRING,
        },
        "product": {
            "type": ExtractParamsSchemaValueType.STRING,
        },
        "magazine": {
            "type": ExtractParamsSchemaValueType.STRING,
        },
        "singer": {
            "type": ExtractParamsSchemaValueType.STRING,
        },
    }

    print("==== Entity Extraction ====")
    print(f"Document: {document}\n")
    print(f"Schema: {schema_}\n")

    print("Extracting entity from document...")
    result = gradient.extract(
        document=document,
        schema_=schema_,
    )
    print(f"Entity: {result['entity']}")
    print("================\n")


def main() -> None:
    gradient = Gradient()

    run_answer_example(gradient=gradient)
    run_summarize_example(gradient=gradient)
    run_analyze_sentiment_example(gradient=gradient)
    run_personalize_example(gradient=gradient)
    run_extract_example(gradient=gradient)

    gradient.close()


if __name__ == "__main__":
    main()
