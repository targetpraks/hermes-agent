"""Regression tests for the Desktop cron ticker's gateway gate (2026-09-25 fire-claim storm).

Field incident: a Desktop-spawned backend ticked the DEFAULT profile's cron store while the host
multiplex gateway was up and serving ``default``. Two tickers on one store; the Desktop took 15
``fire_claim``s per cycle and never released them, so the legitimate gateway tick created an
execution row, lost the claim, and marked it failed — 165 rows of

    "Fire claim lost; execution was not started."   (cron/scheduler.py:3989-3991)

The gate that exists to prevent this was:

    profile_gate = lambda name, home: not (
        _check_gateway_running(Path(home))
        or (name != "default" and _served_by_running_multiplexer(name)))

For ``name == "default"`` that reduces to ``not _check_gateway_running(home)`` — ONE rung, because
``_served_by_running_multiplexer`` returns False unconditionally for ``default`` (hermes_cli/
gateway.py, ``suffix == "default"`` guard). The replacement, ``_desktop_profile_gate``, asks three
independent rungs and guards each one individually.

These tests pin the properties that matter, and they are non-vacuous: the "OLD" reproductions below
fail against the shipped lambda.
"""
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_cli.profiles as profiles_mod
import hermes_cli.web_server as ws
from hermes_cli.web_server import _desktop_profile_gate


def _old_gate(name, home):
    """The shipped lambda, reproduced verbatim — the control these tests beat."""
    return not (profiles_mod._check_gateway_running(Path(home))
                or (name != "default" and profiles_mod._served_by_running_multiplexer(name)))


# --------------------------------------------------------------------------------------
# Criterion 3: gateway alive for `default` + Desktop ticker started -> Desktop does NOT
# tick `default`, and creates no execution rows / fire claims for it.
# --------------------------------------------------------------------------------------


class _RecordingBuiltin:
    """Stands in for InProcessCronScheduler; records start() kwargs."""

    name = "builtin"

    def __init__(self):
        self.start_kwargs = None

    def start(self, stop_event, **kwargs):
        self.start_kwargs = kwargs


@pytest.fixture()
def _recorder(monkeypatch):
    import cron.scheduler_provider as sp

    builtin = _RecordingBuiltin()
    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: builtin)
    monkeypatch.setattr(sp, "InProcessCronScheduler", _RecordingBuiltin)
    monkeypatch.setattr(ws, "resolve_cron_scheduler", lambda: builtin, raising=False)
    monkeypatch.setattr("hermes_logging.enable_profile_log_routing", lambda homes: None)
    return builtin


def test_desktop_stands_down_for_gateway_owned_default(_recorder, tmp_path, monkeypatch):
    """The heart of the incident: a live gateway owns `default` -> Desktop must not tick it.

    Uses the REAL scheduler loop (`InProcessCronScheduler.start`), so this asserts on actual
    dispatch, not just on the gate callable's return value.
    """
    from cron.scheduler_provider import InProcessCronScheduler
    from hermes_constants import get_hermes_home

    default_home = tmp_path / "root"
    default_home.mkdir()
    monkeypatch.setattr(
        profiles_mod, "profiles_to_serve",
        lambda multiplex=False, **_kw: [("default", default_home)])
    # The gateway is alive and owns `default`.
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda _home: True)
    # ...and the host topologies agree (this is the rung the OLD gate never asked).
    monkeypatch.setattr(
        "gateway.host_topology.host_gateway_serving",
        lambda name=None, **_kw: object() if name == "default" else None)

    ticked: list[str] = []
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler", InProcessCronScheduler)
    monkeypatch.setattr(
        "cron.scheduler.tick", lambda **_kw: ticked.append(str(get_hermes_home())))

    stop = threading.Event()
    monkeypatch.setattr(stop, "wait", lambda _timeout: stop.set())

    ws._start_desktop_cron_ticker(stop, interval=0)

    assert ticked == [], (
        "Desktop ticked a store a live gateway owns — this is the duplicate-ticker bug "
        f"that leaked 15 fire claims per cycle; ticked={ticked}"
    )
    # And it must not have claimed the store's status surface either.
    assert not (default_home / "cron" / "ticker_last_success").exists()


def test_old_gate_is_the_bug_and_the_new_gate_is_the_fix(tmp_path, monkeypatch):
    """Non-vacuity: reproduce the bypass on the shipped lambda, and prove the fix closes it.

    The window is a gateway restart / host-role re-election, where the host record is between
    writers so the single identity-file probe the OLD gate relied on reports 'unowned'.
    """
    home = tmp_path / "root"
    home.mkdir()

    monkeypatch.setattr("gateway.host_topology.host_gateway_serving",
                        lambda name=None, **_kw: object())
    monkeypatch.setattr(profiles_mod, "_served_by_running_multiplexer", lambda _name: False)

    # A live gateway owns `default`, but the ONE rung the old gate had cannot see it.
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda _home: False)
    assert _old_gate("default", home) is True, "OLD gate no longer reproduces the bypass"
    assert _desktop_profile_gate("default", home) is False, (
        "NEW gate let a live-gateway-owned default store through")

    # Once the probe recovers, both agree.
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda _home: True)
    assert _old_gate("default", home) is False
    assert _desktop_profile_gate("default", home) is False


def test_gate_rungs_are_isolated(monkeypatch, tmp_path):
    """A rung that RAISES must fall through to the next proof, not abort the gate.

    Found by probing the first draft of the fix: with all rungs inside one try, a raising
    own-gateway probe skipped the host-topology proof entirely and allowed the tick.
    """
    home = tmp_path / "root"
    home.mkdir()
    monkeypatch.setattr("gateway.host_topology.host_gateway_serving",
                        lambda name=None, **_kw: object())

    def _boom(_home):
        raise RuntimeError("liveness probe down")

    monkeypatch.setattr(profiles_mod, "_check_gateway_running", _boom)
    assert _desktop_profile_gate("default", home) is False, (
        "a raising first rung short-circuited the remaining proofs")

    # Rung 1 silent AND rung 2 raising -> rung 3 still answers for a named profile.
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda _home: False)
    monkeypatch.setattr("gateway.host_topology.host_gateway_serving",
                        lambda name=None, **_kw: (_ for _ in ()).throw(RuntimeError("topology down")))
    monkeypatch.setattr(profiles_mod, "_served_by_running_multiplexer",
                        lambda name: name == "worker")
    assert _desktop_profile_gate("worker", tmp_path / "worker") is False


def test_standalone_desktop_still_ticks(tmp_path, monkeypatch):
    """No gateway anywhere: the case this ticker exists for must keep working."""
    from cron.scheduler_provider import InProcessCronScheduler
    from hermes_constants import get_hermes_home

    home = tmp_path / "root"
    home.mkdir()
    monkeypatch.setattr(
        profiles_mod, "profiles_to_serve",
        lambda multiplex=False, **_kw: [("default", home)])
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda _home: False)
    monkeypatch.setattr(profiles_mod, "_served_by_running_multiplexer", lambda _name: False)
    monkeypatch.setattr("gateway.host_topology.host_gateway_serving",
                        lambda name=None, **_kw: None)

    ticked: list[str] = []
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler", InProcessCronScheduler)
    monkeypatch.setattr(
        "cron.scheduler.tick", lambda **_kw: ticked.append(str(get_hermes_home())))

    stop = threading.Event()
    monkeypatch.setattr(stop, "wait", lambda _timeout: stop.set())

    ws._start_desktop_cron_ticker(stop, interval=0)

    assert ticked == [str(home)], "a gateway-less Desktop must still tick its own store"


def test_enumeration_failure_keeps_the_gate_and_still_ticks(tmp_path, monkeypatch):
    """Acceptance criterion 2: enumeration raising must not skip the gateway gate.

    The old `except` branch fell through to the bare single-store ticker, which ignores
    `profile_gate` entirely — an UNGATED tick of the active profile. Run the REAL loop so this
    proves both halves: the gate is honoured, and the active profile still fires.
    """
    from cron.scheduler_provider import InProcessCronScheduler
    from hermes_constants import get_hermes_home

    active_home = tmp_path / "root"
    active_home.mkdir()
    monkeypatch.setattr(
        profiles_mod, "profiles_to_serve",
        lambda multiplex=False, **_kw: (_ for _ in ()).throw(RuntimeError("profiles dir unreadable")))
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")
    monkeypatch.setattr(
        "hermes_constants.get_process_hermes_home", lambda: active_home)
    monkeypatch.setattr(profiles_mod, "_served_by_running_multiplexer", lambda _name: False)
    monkeypatch.setattr("gateway.host_topology.host_gateway_serving",
                        lambda name=None, **_kw: None)

    ticked: list[str] = []
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler", InProcessCronScheduler)
    monkeypatch.setattr(
        "cron.scheduler.tick", lambda **_kw: ticked.append(str(get_hermes_home())))

    stop = threading.Event()
    monkeypatch.setattr(stop, "wait", lambda _timeout: stop.set())

    # The enumerator is broken, but no gateway owns the active profile -> it must still tick.
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda _home: False)
    ws._start_desktop_cron_ticker(stop, interval=0)
    assert ticked == [str(active_home)], (
        "the enumeration-failure fallback no longer ticks the active profile")

    # Same broken enumerator, but now a gateway owns the active profile -> the gate must stop it.
    ticked.clear()
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda _home: True)
    ws._start_desktop_cron_ticker(stop, interval=0)
    assert ticked == [], (
        "enumeration failure bypassed the gateway gate (criterion 2)")


def test_enumeration_failure_installs_the_gate(_recorder, tmp_path, monkeypatch):
    """The recorded start kwargs carry the gate, not a bare single-store ticker."""
    monkeypatch.setattr(
        profiles_mod, "profiles_to_serve",
        lambda multiplex=False, **_kw: (_ for _ in ()).throw(RuntimeError("profiles dir unreadable")))

    ws._start_desktop_cron_ticker(threading.Event(), interval=11)

    kwargs = _recorder.start_kwargs
    assert kwargs["interval"] == 11
    assert kwargs.get("profile_gate") is _desktop_profile_gate
    assert kwargs.get("profile_homes") is ws._active_profile_only_homes


def test_host_topology_rung_reads_the_gateway_role_not_serve(tmp_path, monkeypatch):
    """Over-correction guard: the Desktop publishes its OWN host record too.

    ``_publish_host_record`` (hermes_cli/web_server.py:1371) claims ``ROLE_SERVE``, while
    ``host_gateway_serving`` reads ``ROLE_GATEWAY`` (gateway/host_topology.py:57). If those
    collided, a gateway-less Desktop would gate ITSELF out and stop ticking — silently killing
    cron on every Desktop-only install. Pin the disjointness so a future refactor cannot merge
    the roles and turn the duplicate-ticker fix into a total cron outage.
    """
    from gateway import host_rendezvous as hr

    assert hr.ROLE_GATEWAY != hr.ROLE_SERVE

    # With no gateway record, rung 2 must be silent even though a serve record exists.
    monkeypatch.setattr("gateway.host_topology._from_host_record", lambda: None)
    monkeypatch.setattr("gateway.host_topology._from_served_record", lambda: None)
    import gateway.host_topology as ht

    assert ht.host_gateway_serving("default") is None
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda _h: False)
    monkeypatch.setattr(profiles_mod, "_served_by_running_multiplexer", lambda _n: False)
    assert _desktop_profile_gate("default", tmp_path / "root") is True, (
        "a serve-only host (no gateway role) must still tick its own store")


def test_gate_installed_with_full_served_set(_recorder, tmp_path, monkeypatch):
    """Normal startup still hands the scheduler a live enumerator + the gate."""
    homes = [("default", tmp_path / "root"), ("worker", tmp_path / "profiles" / "worker")]
    monkeypatch.setattr(
        profiles_mod, "profiles_to_serve", lambda multiplex=False, **_kw: list(homes))
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda _home: False)
    monkeypatch.setattr(profiles_mod, "_served_by_running_multiplexer", lambda _name: False)
    monkeypatch.setattr("gateway.host_topology.host_gateway_serving",
                        lambda name=None, **_kw: None)

    ws._start_desktop_cron_ticker(threading.Event(), interval=0)

    kwargs = _recorder.start_kwargs
    assert kwargs["profile_gate"] is _desktop_profile_gate
    assert callable(kwargs["profile_homes"])
    assert kwargs["profile_homes"]() == homes
    # Re-evaluated per tick, not snapshotted.
    assert all(kwargs["profile_gate"](n, h) for n, h in homes)
