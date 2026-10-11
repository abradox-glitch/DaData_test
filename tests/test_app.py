from __future__ import annotations

import json
import unittest

import httpx

from site_finder.app import (
    CandidateEvidence,
    Company,
    LLMDecision,
    OpenAICompatibleJudge,
    OrgStatus,
    PageFetcher,
    SearchResult,
    SiteFinderPipeline,
    is_aggregator,
    normalize_domain,
)


class _DaData:
    def find_party(self, inn: str) -> Company:
        return Company(
            inn=inn,
            kind="LEGAL",
            name_short="ООО Тест",
            name_full="Общество с ограниченной ответственностью Тест",
            status=OrgStatus.ACTIVE,
            city="Москва",
        )


class _Search:
    name = "fake"

    def search(self, query: str, count: int = 10) -> list[SearchResult]:
        return [
            SearchResult(
                url="https://company-example.com/about",
                title="ООО Тест — официальный сайт",
                snippet="Контакты ООО Тест",
                provider=self.name,
                rank=1,
            ),
            SearchResult(
                url="https://www.rusprofile.ru/id/1",
                title="Справочник",
                provider=self.name,
                rank=2,
            ),
        ]


class _Fetcher:
    def fetch(self, url: str) -> tuple[str, str, str] | None:
        return "ООО Тест", "О компании", "ИНН 7721581040 Контакты ООО Тест"


class _Judge:
    def __init__(self, decision: LLMDecision):
        self.decision = decision
        self.calls: list[list[CandidateEvidence]] = []

    def choose_domain(self, company: Company, candidates: list[CandidateEvidence]) -> LLMDecision:
        self.calls.append(candidates)
        return self.decision


class AppTests(unittest.TestCase):
    def _pipeline(self, decision: LLMDecision) -> tuple[SiteFinderPipeline, _Judge]:
        judge = _Judge(decision)
        pipeline = SiteFinderPipeline(
            dadata=_DaData(),  # type: ignore[arg-type]
            search=_Search(),  # type: ignore[arg-type]
            fetcher=_Fetcher(),  # type: ignore[arg-type]
            judge=judge,
        )
        return pipeline, judge

    def test_pipeline_returns_only_a_verified_candidate(self) -> None:
        pipeline, judge = self._pipeline(
            LLMDecision(domain="company-example.com", confidence=0.91, reason="INN and name match")
        )

        result = pipeline.resolve("7721581040")

        self.assertEqual(result.domain, "company-example.com")
        self.assertEqual(result.reason, "verified")
        self.assertEqual([item.domain for item in judge.calls[0]], ["company-example.com"])

    def test_pipeline_rejects_a_hallucinated_domain_and_invalid_inn(self) -> None:
        pipeline, judge = self._pipeline(
            LLMDecision(domain="invented-example.com", confidence=0.99, reason="Looks likely")
        )

        self.assertEqual(pipeline.resolve("7721581040").reason, "llm_domain_not_in_candidates")
        self.assertEqual(pipeline.resolve("7721581041").reason, "invalid_inn")
        self.assertEqual(len(judge.calls), 1)

    def test_url_filters_and_llm_adapter(self) -> None:
        self.assertEqual(normalize_domain("https://shop.Example.co.uk/path"), "example.co.uk")
        self.assertTrue(is_aggregator("https://www.rusprofile.ru/id/1"))
        self.assertFalse(PageFetcher.is_safe_public_url("http://127.0.0.1/admin"))

        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "domain": "company-example.com",
                                        "confidence": 0.88,
                                        "reason": "name and INN match",
                                    }
                                )
                            }
                        }
                    ]
                },
            )

        judge = OpenAICompatibleJudge(
            api_key="test-key",
            base_url="https://llm.example/v1",
            model="test-model",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        company = _DaData().find_party("7721581040")
        decision = judge.choose_domain(
            company,
            [
                CandidateEvidence(
                    domain="company-example.com",
                    source_url="https://company-example.com",
                    source="search",
                )
            ],
        )
        self.assertEqual(captured["url"], "https://llm.example/v1/chat/completions")
        self.assertEqual(decision.domain, "company-example.com")
