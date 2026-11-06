from scripts import serve_policy


def test_create_policy_uses_explicit_task_checkpoint(monkeypatch):
    requested = {}
    result = object()

    def create_trained_policy(config, checkpoint, *, default_prompt):
        requested.update(config=config.name, checkpoint=checkpoint, prompt=default_prompt)
        return result

    monkeypatch.setattr("openpi.policies.policy_config.create_trained_policy", create_trained_policy)
    args = serve_policy.Args(
        config_name="pi05_iPhoneSingle_book_insertion_v3_100_reactive",
        checkpoint_dir="/path/to/checkpoint",
        default_prompt="Insert the book into the shelf.",
    )
    assert serve_policy.create_policy(args) is result
    assert requested == {
        "config": args.config_name,
        "checkpoint": args.checkpoint_dir,
        "prompt": args.default_prompt,
    }
    assert args.host == "127.0.0.1"


def test_server_metadata_announces_the_muscle_vlta_sensor_layout():
    from openpi.training import config as _config

    lift = serve_policy.server_metadata(_config.get_config("pi05_ar3_lift_reactive"))
    assert lift["sensor_history"] == 0
    assert lift["tactile_in_use"] is False
    muscle = serve_policy.server_metadata(_config.get_config("pi05_ar3_muscle_vlta"))
    assert muscle["sensor_history"] == 10
    assert muscle["tactile_in_use"] is True
    assert muscle["wrench_dim"] == 13
