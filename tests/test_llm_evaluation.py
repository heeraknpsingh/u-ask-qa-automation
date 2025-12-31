"""
Tests for LLM Response Evaluation using RAGAS, DeepEval, and HuggingFace.
These tests evaluate chatbot responses for:
- Hallucination detection
- Context relevancy
- Answer correctness
- Toxicity and bias
Requires: pip install -e '.[eval]'
"""
# pylint: disable=import-error, too-few-public-methods, too-many-locals

import logging

import pytest
from u_ask_qa.pages import ChatbotPage

logger = logging.getLogger(__name__)


class TestLLMEvaluation:
    """LLM-powered response evaluation tests."""

    @pytest.mark.llm_eval
    @pytest.mark.response
    async def test_response_faithfulness(
        self, chatbot_page: ChatbotPage, llm_evaluator, test_data
    ):
        """
        Test that chatbot response is faithful to the retrieved context.
        Detects hallucinations where the response contains claims not in context.
        """
        # Prefer a curated evaluation case with ground-truth + optional context.
        eval_cases = test_data.get_evaluation_cases_as_dict(lang="en")
        if eval_cases:
            case = eval_cases[0]
            question = case["question"]
            ground_truth = case.get("ground_truth")
        else:
            # Fallback to smoke prompt when evaluation set isn't present.
            prompts = test_data.get_smoke_prompts()
            prompt = next((p for p in prompts if p.get("lang") == "en"), prompts[0])
            question = prompt["q"]
            ground_truth = None

        # Send message and get response
        await chatbot_page.send_message(question)
        await chatbot_page.wait_for_response()
        responses = await chatbot_page.get_all_responses()

        assert responses, "No response received from bot"
        response = responses[-1]

        # Get response with context (handles source extraction and fetching)
        result_data = await chatbot_page.get_response_with_context()
        
        # Log source fetch statistics
        source_stats = result_data.get("source_stats", {})
        logger.info(
            "Source fetch stats: total=%d fetched=%d failed=%d coverage=%.1f%%",
            source_stats.get("total", 0),
            source_stats.get("fetched", 0),
            source_stats.get("failed", 0),
            source_stats.get("coverage", 0) * 100,
        )
        
        # Log any failed sources
        for fetch_result in result_data.get("fetch_results", []):
            if not fetch_result.get("success"):
                logger.warning(
                    "Failed to fetch source: %s | error: %s",
                    fetch_result.get("url"),
                    fetch_result.get("error"),
                )

        # Get context from the result
        context = result_data.get("context", [])
        
        # IMPORTANT: context must come from citation websites only.
        # If the UI provides no citations or scraping fails, fail the test.
        assert context and context[0], (
            "No citation-based context could be built from Sources/References. "
            f"Source stats: {source_stats}"
        )

        # Evaluate with configured evaluator backend.
        result = await llm_evaluator.evaluate(
            question=question,
            response=response,
            context=context,
            ground_truth=ground_truth,
        )

        # Log evaluation results
        logger.info("\n%s", "=" * 60)
        logger.info("Question: %s", question)
        logger.info("Response: %s...", response)
        logger.info("References found: %d", len(result_data.get("sources", [])))
        logger.info("Context %s", context)
        logger.info("Context chunks: %d", len(context))
        logger.info("Result: %s", result)
        logger.info("Overall Score: %.2f", result.overall_score)
        logger.info("Is Passing: %s", result.is_passing)
        if result.issues:
            logger.info("Issues: %s", result.issues)
        logger.info("%s\n", "=" * 60)

        # Assert evaluation passed
        assert result.overall_score >= 0.5, (
            f"Response quality too low: {result.overall_score:.2f}. "
            f"Issues: {', '.join(result.issues)}"
        )


class TestSourceFetching:
    """Tests for verifying source fetching works correctly."""

    @pytest.mark.llm_eval
    @pytest.mark.response
    async def test_source_extraction_and_fetching(
        self, chatbot_page: ChatbotPage, test_data
    ):
        """
        Test that sources are correctly extracted and fetched from the chatbot.
        """
        # Use a question that typically returns sources
        prompts = test_data.get_smoke_prompts()
        prompt = next((p for p in prompts if p.get("lang") == "en"), prompts[0])
        question = prompt["q"]

        # Send message and wait for response
        await chatbot_page.send_message(question)
        await chatbot_page.wait_for_response()

        # Get response with context
        result_data = await chatbot_page.get_response_with_context()

        # Check that we got a response
        assert result_data.get("response"), "No response received"

        # Check source stats
        source_stats = result_data.get("source_stats", {})
        total_sources = source_stats.get("total", 0)
        fetched_sources = source_stats.get("fetched", 0)
        coverage = source_stats.get("coverage", 0)

        logger.info("Source extraction results:")
        logger.info("  Total sources found: %d", total_sources)
        logger.info("  Successfully fetched: %d", fetched_sources)
        logger.info("  Coverage: %.1f%%", coverage * 100)

        # Log individual fetch results
        for fetch_result in result_data.get("fetch_results", []):
            status = "✅" if fetch_result.get("success") else "❌"
            logger.info(
                "  %s %s (%d chars)",
                status,
                fetch_result.get("url", "unknown"),
                fetch_result.get("chars", 0),
            )

        # Assert minimum requirements
        assert total_sources > 0, "No sources found in response"
        assert fetched_sources > 0, "No sources could be fetched"
        assert coverage >= 0.5, f"Source coverage too low: {coverage:.1%}"


#     @pytest.mark.llm_eval
#     @pytest.mark.response
#     async def test_response_relevancy(
#         self, chatbot_page: ChatbotPage, llm_evaluator
#     ):
#         """
#         Test that chatbot response is relevant to the question asked.
#         """
#         question = "What documents are needed for a UAE tourist visa?"

#         await chatbot_page.send_message(question)
#         await chatbot_page.wait_for_response()
#         responses = await chatbot_page.get_all_responses()

#         assert responses, "No response received from bot"
#         response = responses[-1]

#         result = await llm_evaluator.evaluate(
#             question=question,
#             response=response,
#         )

#         # Check answer relevancy specifically
#         if result.metrics.answer_relevancy:
#             relevancy_score = result.metrics.answer_relevancy.score
#             print(f"Answer Relevancy Score: {relevancy_score:.2f}")
#             assert relevancy_score >= 0.6, (
#                 f"Response not relevant enough: {relevancy_score:.2f}"
#             )

#     @pytest.mark.llm_eval
#     @pytest.mark.response
#     async def test_response_against_ground_truth(
#         self, chatbot_page: ChatbotPage, llm_evaluator
#     ):
#         """
#         Test response accuracy against known ground truth answer.
#         Uses semantic similarity to compare responses.
#         """
#         question = "What is the emergency number in UAE?"
#         ground_truth = (
#             "The emergency number in UAE is 999 for police, 998 for ambulance, "
#             "and 997 for fire."
#         )

#         await chatbot_page.send_message(question)
#         await chatbot_page.wait_for_response()
#         responses = await chatbot_page.get_all_responses()

#         assert responses, "No response received from bot"
#         response = responses[-1]

#         result = await llm_evaluator.evaluate(
#             question=question,
#             response=response,
#             ground_truth=ground_truth,
#         )

#         # Check semantic similarity to ground truth
#         if result.metrics.semantic_similarity:
#             similarity = result.metrics.semantic_similarity.score
#             print(f"Semantic Similarity to Ground Truth: {similarity:.2f}")
#             assert similarity >= 0.5, (
#                 f"Response doesn't match expected answer: {similarity:.2f}"
#             )

#     @pytest.mark.llm_eval
#     @pytest.mark.response
#     @pytest.mark.security
#     async def test_response_toxicity(
#         self, chatbot_page: ChatbotPage, llm_evaluator
#     ):
#         """
#         Test that chatbot responses are not toxic or harmful.
#         """
#         # Test with a normal question
#         question = "Tell me about UAE government services"

#         await chatbot_page.send_message(question)
#         await chatbot_page.wait_for_response()
#         responses = await chatbot_page.get_all_responses()

#         assert responses, "No response received from bot"
#         response = responses[-1]

#         result = await llm_evaluator.evaluate(
#             question=question,
#             response=response,
#         )

#         # Check toxicity metric
#         if result.metrics.toxicity:
#             toxicity_score = result.metrics.toxicity.score
#             print(f"Toxicity Score: {toxicity_score:.2f}")
#             # Score is inverted: higher = less toxic
#             assert toxicity_score >= 0.7, (
#                 f"Response may contain toxic content. "
#                 f"Toxicity check score: {toxicity_score:.2f}"
#             )

#     @pytest.mark.llm_eval
#     @pytest.mark.response
#     async def test_arabic_response_quality(
#         self, chatbot_page_ar: ChatbotPage, llm_evaluator, test_data
#     ):
#         """
#         Test Arabic response quality using LLM evaluation.
#         """
#         prompts = test_data.get_arabic_prompts()
#         if not prompts:
#             pytest.skip("No Arabic prompts available")

#         prompt = prompts[0]
#         question = prompt["q"]

#         await chatbot_page_ar.send_message(question)
#         await chatbot_page_ar.wait_for_response()
#         responses = await chatbot_page_ar.get_all_responses()

#         assert responses, "No response received from bot"
#         response = responses[-1]

#         result = await llm_evaluator.evaluate(
#             question=question,
#             response=response,
#         )

#         print(f"\n{'='*60}")
#         print(f"Arabic Question: {question}")
#         print(f"Response: {response[:200]}...")
#         print(f"Overall Score: {result.overall_score:.2f}")
#         print(f"{'='*60}\n")

#         assert result.overall_score >= 0.4, (
#             f"Arabic response quality too low: {result.overall_score:.2f}"
#         )

#     @pytest.mark.llm_eval
#     @pytest.mark.response
#     async def test_hallucination_on_specific_query(
#         self, chatbot_page: ChatbotPage, llm_evaluator
#     ):
#         """
#         Test hallucination detection on a query with known facts.
#         """
#         question = "What are the working hours of UAE government offices?"
#         # Known context about UAE government working hours
#         known_context = [
#             "UAE government offices typically operate Sunday to Thursday.",
#             "Standard working hours are from 7:30 AM to 3:30 PM.",
#             "Friday and Saturday are the weekend in UAE.",
#         ]

#         await chatbot_page.send_message(question)
#         await chatbot_page.wait_for_response()
#         responses = await chatbot_page.get_all_responses()

#         assert responses, "No response received from bot"
#         response = responses[-1]

#         result = await llm_evaluator.evaluate(
#             question=question,
#             response=response,
#             context=known_context,
#         )

#         # Check hallucination metric
#         if result.metrics.hallucination:
#             hallucination_score = result.metrics.hallucination.score
#             print(f"Hallucination Score: {hallucination_score:.2f}")
#             assert hallucination_score >= 0.5, (
#                 f"Response may contain hallucinated information. "
#                 f"Score: {hallucination_score:.2f}"
#             )

#         # Also check faithfulness if available
#         if result.metrics.faithfulness:
#             faithfulness_score = result.metrics.faithfulness.score
#             print(f"Faithfulness Score: {faithfulness_score:.2f}")
#             assert faithfulness_score >= 0.5, (
#                 f"Response not faithful to context. Score: {faithfulness_score:.2f}"
#             )


# class TestLLMEvaluationMetrics:
#     """Tests for verifying evaluation metrics are working correctly."""

#     @pytest.mark.llm_eval
#     def test_evaluator_backends_available(self, llm_evaluator):
#         """Verify which evaluation backends are installed."""
#         backends = llm_evaluator.available_backends
#         print(f"Available backends: {backends}")
#         assert len(backends) > 0, "No evaluation backends available"

#     @pytest.mark.llm_eval
#     def test_evaluator_config_thresholds(self, evaluator_config):
#         """Verify evaluation thresholds are properly configured."""
#         thresholds = evaluator_config.get_thresholds()
#         print(f"Configured thresholds: {thresholds}")

#         # Verify all thresholds are valid
#         for name, value in thresholds.items():
#             assert 0.0 <= value <= 1.0, f"Invalid threshold for {name}: {value}"

#     @pytest.mark.llm_eval
#     async def test_basic_evaluation_flow(self, llm_evaluator):
#         """Test basic evaluation without browser interaction."""
#         result = await llm_evaluator.evaluate(
#             question="What is the capital of UAE?",
#             response="The capital of the United Arab Emirates is Abu Dhabi.",
#             ground_truth="Abu Dhabi is the capital city of the UAE.",
#         )

#         print(f"Evaluation Result: {result.to_dict()}")
#         assert result.overall_score >= 0, "Invalid overall score"
#         assert isinstance(result.is_passing, bool), "is_passing should be boolean"


# class TestLLMEvaluationWithTestData:
#     """Tests using prompts from test-data.json with LLM evaluation."""

#     @pytest.mark.llm_eval
#     @pytest.mark.response
#     @pytest.mark.parametrize("prompt_tag", ["visa", "id", "driving"])
#     async def test_category_responses(
#         self, chatbot_page: ChatbotPage, llm_evaluator, test_data, prompt_tag
#     ):
#         """
#         Test responses for different prompt categories.
#         Parameterized to run for multiple categories.
#         """
#         prompts = [
#             p for p in test_data.get_english_prompts()
#             if prompt_tag in p.get("tags", [])
#         ]

#         if not prompts:
#             pytest.skip(f"No prompts found for category: {prompt_tag}")

#         # Test first prompt from category
#         prompt = prompts[0]
#         question = prompt["q"]

#         await chatbot_page.send_message(question)
#         await chatbot_page.wait_for_response()
#         responses = await chatbot_page.get_all_responses()

#         assert responses, f"No response for {prompt_tag} query"
#         response = responses[-1]

#         result = await llm_evaluator.evaluate(
#             question=question,
#             response=response,
#         )

#         print(f"\nCategory: {prompt_tag}")
#         print(f"Question: {question}")
#         print(f"Score: {result.overall_score:.2f}")

#         assert result.overall_score >= 0.4, (
#             f"Poor quality response for {prompt_tag}: {result.overall_score:.2f}"
#         )
