from handgen.eval.harness import _carrier_prompt_policy_from_summary, summarize_manifest


def test_carrier_prompt_policy_keeps_placeholder_specific_rankings():
    summary = [
        {
            "context": "then_slot_clause",
            "placeholder": "term",
            "prompt_template": "Then {placeholder} is positive",
            "mean_match_score": 0.4,
            "exact_count": 0,
            "last_resort_count": 0,
        },
        {
            "context": "then_slot_clause",
            "placeholder": "term",
            "prompt_template": "Then {placeholder} is useful",
            "mean_match_score": 0.2,
            "exact_count": 0,
            "last_resort_count": 0,
        },
        {
            "context": "then_slot_clause",
            "placeholder": "number",
            "prompt_template": "Then {placeholder} is useful",
            "mean_match_score": 0.5,
            "exact_count": 0,
            "last_resort_count": 0,
        },
    ]

    policy = _carrier_prompt_policy_from_summary(summary, top_k=1)

    context = policy["contexts"]["then_slot_clause"]
    assert context["prompt_templates_by_placeholder"]["term"] == ["Then {placeholder} is positive"]
    assert context["prompt_templates_by_placeholder"]["number"] == ["Then {placeholder} is useful"]
    assert context["ranked_prompt_templates_by_placeholder"]["term"][0]["placeholder"] == "term"


def test_summarize_manifest_reports_quality_gates_and_slot_detection():
    manifest = {
        "source_contract": {"passed": True},
        "routes_used": ["diffbrush_best_effort_prose_token", "worksheet_fragment_replaces_diffbrush_blank_slot"],
        "warnings": [],
        "spans": [
            {
                "type": "slot_carrier_prose",
                "route": "diffbrush_best_effort_prose_token",
                "diffbrush": {
                    "ocr_verification": {"match_score": 0.42},
                    "best_effort_effective_min_score": 0.3,
                    "exactness_certified": False,
                },
            },
            {
                "id": "slot",
                "type": "math_slot_replacement",
                "text": "x + y",
                "bbox": {"width": 40, "height": 20},
                "slot_geometry": {"target_visible_height": 18},
                "slot_word_box_source": "visual_alpha_refined_from_estimated_from_slot_carrier_text",
                "slot_word_box": {"visual_detection": {"method": "alpha_column_run_overlap"}},
            },
        ],
        "carrier_plan": [
            {
                "insertions": [
                    {"text": "x + y", "slot_placeholder": {"text": "number"}},
                ],
                "source_text": "If x + y is true.",
            }
        ],
        "outputs": {"png": "document.png"},
    }

    metrics = summarize_manifest(manifest)

    assert metrics["quality_gates"]["passed"] is True
    assert metrics["slot_visual_detection_counts"] == {"alpha_column_run_overlap": 1}
    assert metrics["slot_word_box_source_counts"] == {"visual_alpha_refined_from_estimated_from_slot_carrier_text": 1}
    assert metrics["slot_sizes"][0]["visual_detection_method"] == "alpha_column_run_overlap"
