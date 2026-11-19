from physics.verify import verify_candidates
from pipeline.controller import analyze


def test_callback_and_callback_failure(
    prepared, mock_recon, observation, isolated_runs, stub_config, stub_observation_stages,
):
    calls = []

    def subscriber(fraction, message):
        calls.append((fraction, message))
        raise RuntimeError("UI disconnected")

    for invoke in (lambda callback: verify_candidates(prepared, mock_recon, {}, progress=callback),
                   lambda callback: analyze(observation, stub_config, progress=callback)):
        calls.clear()
        invoke(None).validate()
        invoke(subscriber).validate()
        fractions = [fraction for fraction, _ in calls]
        assert fractions == sorted(fractions)
        assert fractions[0] == 0.0 and fractions[-1] == 1.0
        assert all(0 <= fraction <= 1 and type(fraction) is float for fraction in fractions)
        assert all(isinstance(message, str) and message.strip() and "\n" not in message
                   for _, message in calls)
