from refvision.tracking import Detection, DirectionalLineCounter, MajorityVote, TrackAggregator


def test_majority_vote_stabilizes_label():
    vote = MajorityVote(window=5, min_votes=3)
    for label in ["apple::normal", "apple::damaged", "apple::normal", "apple::normal"]:
        vote.update(7, label, 0.8)
    stable, confidence = vote.stable(7)
    assert stable == "apple::normal"
    assert confidence > 0


def test_directional_line_counts_once():
    counter = DirectionalLineCounter(
        [[0.0, 0.5], [1.0, 0.5]],
        direction="any",
        count_once_per_track=True,
    )
    assert counter.update(3, (50, 40), (100, 100)) is None
    assert counter.update(3, (50, 60), (100, 100)) is not None
    assert counter.update(3, (50, 40), (100, 100)) is None


def test_pending_crossing_waits_for_stable_vote():
    agg = TrackAggregator(
        line_norm=[[0.0, 0.5], [1.0, 0.5]],
        direction="any",
        vote_window=5,
        min_votes=3,
        count_once_per_track=True,
    )

    d1 = Detection(1, 0, 0.9, (40, 30, 60, 50))  # centroid y=40
    d2 = Detection(1, 0, 0.9, (40, 50, 60, 70))  # centroid y=60 -> crossing
    d3 = Detection(1, 0, 0.9, (40, 55, 60, 75))

    assert agg.process(1, (100, 100), [(d1, "apple::normal")]) == []
    assert agg.process(2, (100, 100), [(d2, "apple::normal")]) == []
    events = agg.process(3, (100, 100), [(d3, "apple::normal")])
    assert len(events) == 1
    assert events[0].state_name == "normal"
    assert agg.counts["apple::normal"] == 1
