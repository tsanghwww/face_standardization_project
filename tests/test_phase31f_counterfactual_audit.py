"""CPU protocol checks for Phase3.1f counterfactual output tracking."""

from phase3.counterfactual_geometry import counterfactual_tracking_metrics, summarize_counterfactual_rows


def main() -> None:
    negative_target = [0.0, -0.1745329252, 0.0]
    positive_target = [0.0, 0.1745329252, 0.0]
    correct = counterfactual_tracking_metrics(
        [0.0, -0.10, 0.0], [0.0, 0.10, 0.0], negative_target, positive_target
    )
    assert correct["direction_correct"] and correct["ordering_correct"]
    assert correct["projected_output_change_deg"] > 0
    reversed_result = counterfactual_tracking_metrics(
        [0.0, 0.10, 0.0], [0.0, -0.10, 0.0], negative_target, positive_target
    )
    assert not reversed_result["direction_correct"] and not reversed_result["ordering_correct"]

    rows = []
    for image_id in ("a", "b"):
        for timestep in (100, 400):
            rows.append({"image_id": image_id, "timestep": timestep, "status": "success", **correct})
    report = summarize_counterfactual_rows(rows, ["a", "b"], [100, 400])
    assert report["overall"]["n_total"] == 4
    assert report["overall"]["both_correct"] == 4
    assert report["overall"]["candidate_gate"]["passed"]
    print("Phase3.1f counterfactual output direction/order protocol passed")


if __name__ == "__main__":
    main()
