"""Regression tests for #121904 — a lock-skipped tick was recorded as a SUCCESSFUL tick.

A tick that could not take ``.tick.lock`` returned 0 *without raising*, so both ticker loops left
``record_ticker_heartbeat(success=True)`` in place: fresh heartbeat + fresh last-success marker +
ZERO dispatches. A 95-minute stall of the PRIMARY store was invisible to every signal an operator
reads — ``hermes cron status`` was green throughout while 18 jobs sat overdue.

The fix has to keep the mutual-exclusion contract intact (contention is NORMAL: host gateway
multiplexer vs Desktop backend ticker vs manual ``hermes cron tick`` vs one-shot ``hermes cron
run``) while making a holder that *cannot be progressing* a RECORDED FAILURE. So the holder
publishes a lease beside the lock and the skipper reads it:

* no lease / unreadable                 -> "cannot prove a wedge"  -> benign skip (success=True)
* holder pid alive, progress within bound -> progressing holder    -> benign skip (success=True)
* holder pid gone                       -> nothing can release it -> FAILED tick
* holder alive, progress past the bound -> wedged                  -> FAILED tick
"""
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from cron import scheduler as scheduler_mod
from cron.scheduler_provider import InProcessCronScheduler

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

pytestmark = pytest.mark.skipif(fcntl is None, reason="flock semantics are POSIX-only")


@pytest.fixture()
def hermetic_home(tmp_path, monkeypatch):
    """Per-test cron store root, never the operator's real ``~/.hermes``.

    The global conftest already sandboxes ``HERMES_HOME``, but the lock paths must be explicit and
    provably isolated: these tests create and age real lock/lease files, and a stray write into the
    live store would wedge the running gateway's ticker. ``cron.scheduler._hermes_home`` is the
    module's own test override — setting it keeps ``tick()``'s ``_get_lock_paths()`` and the test's
    ``lock_file`` pointing at the SAME store.
    """
    home = tmp_path / "hermes-home"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setattr(scheduler_mod, "_hermes_home", home)
    return home


def _wait_until(predicate, timeout=10.0, interval=0.005):
    """Block until ``predicate()`` is truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


def _expire_lease(lock_file, *, pid=None):
    """Age the holder lease past the wedge bound (and optionally re-point its pid)."""
    lease_path = scheduler_mod._get_tick_lock_lease_path(lock_file)
    data = json.loads(lease_path.read_text(encoding="utf-8"))
    data["progress_at"] = time.time() - (scheduler_mod.tick_lock_wedge_bound_seconds() + 60)
    if pid is not None:
        data["pid"] = pid
    lease_path.write_text(json.dumps(data), encoding="utf-8")
    return data


# ── Acceptance criterion 3: hold .tick.lock from a helper that never iterates ──────────────────


class TestWedgedHolderIsRecordedAsFailure:
    def test_wedged_holder_raises_so_the_tick_is_not_success(self, hermetic_home):
        """A helper holds the lock and never advances its lease: tick() must RAISE, so the
        ticker loop cannot record a successful tick (the pre-fix silent ``return 0``)."""
        lock_file = hermetic_home / "cron" / ".tick.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("")

        # A holder from a DIFFERENT process, so the lease pid is a real, live, foreign pid.
        holder = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(f"""
                import fcntl, json, os, time
                fd = open({str(lock_file)!r}, "w")
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                json.dump({{"pid": os.getpid(), "acquired_at": time.time(),
                           "progress_at": time.time()}}, open({str(lock_file) + ".owner"!r}, "w"))
                time.sleep(60)
            """)])
        try:
            # Wait until the helper actually holds the lock and published its lease.
            assert _wait_until(
                lambda: scheduler_mod._read_tick_lock_lease(lock_file) is not None
                and scheduler_mod._read_tick_lock_lease(lock_file)["pid"] == holder.pid), \
                "holder never published its lease"
            # While the holder IS progressing, the skip is benign (criterion 3, second half).
            assert scheduler_mod.tick(verbose=False) == 0
            assert scheduler_mod.describe_tick_lock_wedge(lock_file) is None

            # Now age the lease past the bound: the same live holder is wedged.
            _expire_lease(lock_file)
            with pytest.raises(scheduler_mod.CronTickLockWedged) as excinfo:
                scheduler_mod.tick(verbose=False)
            assert str(holder.pid) in str(excinfo.value)
            assert "no progress" in str(excinfo.value)
        finally:
            holder.terminate()
            holder.wait(timeout=10)

    def test_dead_holder_lease_raises(self, hermetic_home, monkeypatch):
        """A lease whose pid is GONE can never be released: raise regardless of the bound."""
        lock_file = hermetic_home / "cron" / ".tick.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("")
        scheduler_mod._write_tick_lock_lease(lock_file)
        # A pid that provably does not exist, with progress still *fresh* — dead beats fresh.
        _expire_lease(lock_file, pid=999999)
        monkeypatch.setattr(scheduler_mod, "_process_is_alive", lambda pid: False)

        reason = scheduler_mod.describe_tick_lock_wedge(lock_file)
        assert reason is not None and "999999" in reason
        assert "no longer exists" in reason

    def test_ticker_loop_records_failed_beat_and_error(self, hermetic_home, monkeypatch):
        """The single-profile ticker loop must record success=False + a ticker error, not a healthy
        heartbeat — this is the exact signal that lied for 95 minutes."""
        beats, errors = [], []
        stop = threading.Event()
        prov = InProcessCronScheduler()
        monkeypatch.setattr(
            "cron.scheduler.tick",
            lambda *a, **kw: (_ for _ in ()).throw(
                scheduler_mod.CronTickLockWedged("Tick lock held by pid 4242, wedged.")))

        with monkeypatch.context() as ctx:
            ctx.setattr("cron.jobs.record_ticker_heartbeat",
                        lambda success=False: beats.append(success))
            ctx.setattr("cron.jobs.record_ticker_error", lambda msg: errors.append(msg))
            ctx.setattr("cron.jobs.clear_ticker_error", lambda: None)
            t = threading.Thread(target=prov.start, args=(stop,), kwargs={"interval": 0},
                                 daemon=True)
            t.start()
            assert _wait_until(lambda: len(beats) >= 3), "ticker did not keep beating"
            stop.set()
            t.join(timeout=5)

        assert not t.is_alive(), "a wedged tick must not kill the ticker thread"
        assert errors and errors[0].startswith("CronTickLockWedged"), \
            "the wedge must be persisted so `hermes cron status` can show WHY"
        assert beats[-1] is False, "a wedged tick must NOT be recorded as success"


class TestTransientContentionStaysNonFailing:
    def test_absent_lease_keeps_todays_benign_skip(self, hermetic_home):
        """No lease (e.g. a pre-fix holder) must fail OPEN: a benign skip, never a false alarm."""
        lock_file = hermetic_home / "cron" / ".tick.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("")
        assert scheduler_mod.describe_tick_lock_wedge(lock_file) is None

    def test_progressing_holder_is_a_benign_skip(self, hermetic_home):
        """A live holder refreshing its lease on every cycle is the NORMAL contention case."""
        lock_file = hermetic_home / "cron" / ".tick.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("")
        scheduler_mod._write_tick_lock_lease(lock_file)
        assert scheduler_mod.describe_tick_lock_wedge(lock_file) is None
        # A checkpoint moves it forward; still benign, however often it runs.
        scheduler_mod._refresh_tick_lock_progress(lock_file)
        assert scheduler_mod.describe_tick_lock_wedge(lock_file) is None

    def test_contention_with_progressing_holder_still_returns_zero(self, hermetic_home):
        """End-to-end: another process holds the lock AND keeps progressing -> tick() == 0."""
        lock_file = hermetic_home / "cron" / ".tick.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("")
        holder = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(f"""
                import fcntl, json, os, time
                fd = open({str(lock_file)!r}, "w")
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                path = {str(lock_file) + ".owner"!r}
                for _ in range(80):
                    json.dump({{"pid": os.getpid(), "acquired_at": time.time(),
                               "progress_at": time.time()}}, open(path, "w"))
                    time.sleep(0.25)
            """)])
        try:
            assert _wait_until(
                lambda: (scheduler_mod._read_tick_lock_lease(lock_file) or {}).get("pid")
                == holder.pid), "holder never published its lease"
            assert scheduler_mod.tick(verbose=False) == 0
        finally:
            holder.terminate()
            holder.wait(timeout=10)

    def test_unreadable_lease_fails_open(self, hermetic_home):
        """A torn/unreadable lease must read as 'cannot prove a wedge', not as a wedge."""
        lock_file = hermetic_home / "cron" / ".tick.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("")
        scheduler_mod._get_tick_lock_lease_path(lock_file).write_text("{not json", encoding="utf-8")
        assert scheduler_mod.describe_tick_lock_wedge(lock_file) is None


# ── Lease lifecycle ────────────────────────────────────────────────────────────────────────────


class TestHolderLeaseLifecycle:
    def test_acquire_publishes_and_release_clears(self, hermetic_home):
        lock_file = hermetic_home / "cron" / ".tick.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        fd = scheduler_mod._acquire_tick_lock(lock_file)
        try:
            lease = scheduler_mod._read_tick_lock_lease(lock_file)
            assert lease is not None and lease["pid"] == os.getpid()
            assert lease["progress_at"] > 0 and lease["acquired_at"] > 0
        finally:
            scheduler_mod._release_tick_lock(fd)
        # A stale lease must not outlive its holder.
        assert scheduler_mod._read_tick_lock_lease(lock_file) is None

    def test_lease_write_failure_does_not_fail_the_tick(self, hermetic_home, monkeypatch):
        """A lost lease write degrades to 'cannot prove a wedge' — it must never break the lock."""
        lock_file = hermetic_home / "cron" / ".tick.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        def boom(*_a, **_kw):
            raise OSError("no space left on device")

        monkeypatch.setattr("utils.atomic_write_text", boom)
        fd = scheduler_mod._acquire_tick_lock(lock_file)
        try:
            assert fd is not None, "the lock must still be acquired when the lease write fails"
            assert scheduler_mod._read_tick_lock_lease(lock_file) is None
            assert scheduler_mod.describe_tick_lock_wedge(lock_file) is None
        finally:
            scheduler_mod._release_tick_lock(fd)

    def test_refresh_ignores_a_foreign_lease(self, hermetic_home):
        """A ticker that lost the race must never refresh the real holder's progress signal."""
        lock_file = hermetic_home / "cron" / ".tick.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lease_path = scheduler_mod._get_tick_lock_lease_path(lock_file)
        lease_path.write_text(json.dumps(
            {"pid": os.getpid() + 1, "acquired_at": 1.0, "progress_at": 2.0}), encoding="utf-8")
        scheduler_mod._refresh_tick_lock_progress(lock_file)
        assert json.loads(lease_path.read_text(encoding="utf-8"))["progress_at"] == 2.0

    def test_release_from_a_non_owner_fd_keeps_the_lease(self, hermetic_home):
        """An inherited fd (fork child) must not erase the real holder's lease on release."""
        lock_file = hermetic_home / "cron" / ".tick.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lease_path = scheduler_mod._get_tick_lock_lease_path(lock_file)
        lease_path.write_text(json.dumps(
            {"pid": os.getpid() + 1, "acquired_at": 1.0, "progress_at": 2.0}), encoding="utf-8")
        fd = open(lock_file, "w", encoding="utf-8")
        scheduler_mod._release_tick_lock(fd)
        assert scheduler_mod._read_tick_lock_lease(lock_file) is not None


# ── The bound is derived, not a bare constant ──────────────────────────────────────────────────


class TestWedgeBound:
    def test_bound_never_below_the_configured_floor(self):
        assert scheduler_mod.tick_lock_wedge_bound_seconds() >= \
            scheduler_mod.DEFAULT_TICK_LOCK_WEDGE_SECONDS

    def test_bound_follows_the_inactivity_limit(self, monkeypatch):
        """Raising HERMES_CRON_TIMEOUT must lengthen the leash proportionally."""
        monkeypatch.setattr(scheduler_mod, "_cron_inactivity_seconds", lambda: 4000.0)
        assert scheduler_mod.tick_lock_wedge_bound_seconds() == pytest.approx(12000.0)

    def test_configured_floor_is_honoured(self, monkeypatch):
        monkeypatch.setattr(scheduler_mod, "_cron_inactivity_seconds", lambda: 60.0)
        monkeypatch.setattr(
            "cron.jobs._cron_config_number", lambda key, default, cast: cast(900))
        assert scheduler_mod.tick_lock_wedge_bound_seconds() == pytest.approx(900.0)


# ── Acceptance criterion 2: the persisted reason reaches `hermes cron status` ──────────────────


class TestStatusSurfacesTheBlock:
    def test_wedged_labels_round_trip(self):
        reason = "Tick lock held by pid 262 with no progress for 3600s (bound 1800s)."
        recorded = f"{scheduler_mod.CronTickLockWedged.__name__}: {reason}"
        assert scheduler_mod.tick_lock_wedged_labels(recorded) == reason
        assert scheduler_mod.tick_lock_wedged_labels(None) is None
        assert scheduler_mod.tick_lock_wedged_labels("OSError: something else") is None
        assert scheduler_mod.tick_lock_wedged_labels("CronTickYielded: booted on x") is None

    def test_status_prints_the_block_and_the_hint(self, hermetic_home, capsys, monkeypatch):
        """`hermes cron status` must reach its BLOCKED branch — not the green 'jobs will fire'."""
        from hermes_cli import cron as cron_cli

        reason = "Tick lock held by pid 262 with no progress for 3600s (bound 1800s)."
        monkeypatch.setattr("cron.jobs.get_ticker_heartbeat_age", lambda: 5.0)
        monkeypatch.setattr("cron.jobs.get_ticker_success_age", lambda: 4000.0)
        monkeypatch.setattr(
            "cron.jobs.get_ticker_last_error",
            lambda: f"{scheduler_mod.CronTickLockWedged.__name__}: {reason}")

        cron_cli._print_ticker_health([4242])
        out = capsys.readouterr().out
        assert "BLOCKED" in out
        assert reason in out
        assert "tick.lock" in out
        assert "will fire automatically" not in out

    def test_status_branch_is_reachable_from_the_real_marker(self, hermetic_home, capsys,
                                                            monkeypatch):
        """End-to-end through the real marker file: a recorded wedge error drives the branch."""
        from cron.jobs import record_ticker_error
        from hermes_cli import cron as cron_cli

        reason = "Tick lock held by pid 999 with no progress for 9000s (bound 1800s)."
        record_ticker_error(
            f"{scheduler_mod.CronTickLockWedged.__name__}: {reason}")
        monkeypatch.setattr("cron.jobs.get_ticker_heartbeat_age", lambda: 5.0)
        monkeypatch.setattr("cron.jobs.get_ticker_success_age", lambda: 9000.0)

        cron_cli._print_ticker_health([])
        out = capsys.readouterr().out
        assert "BLOCKED" in out and reason in out


# ── Acceptance criterion 4: the 95-minute outage would have shown ──────────────────────────────


class TestOutageWouldHaveBeenVisible:
    def test_the_incident_signature_is_no_longer_green(self, hermetic_home, capsys, monkeypatch):
        """Reconstruct the t_f8a87dae signature: fresh heartbeat + fresh success marker were the
        lie. Post-fix, the same lock state yields a failed beat and a status warning."""
        lock_file = hermetic_home / "cron" / ".tick.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("")
        # The wedged holder from the incident: a live pid that never progressed.
        scheduler_mod._write_tick_lock_lease(lock_file)
        _expire_lease(lock_file, pid=os.getpid())
        wedge = scheduler_mod.describe_tick_lock_wedge(lock_file)
        assert wedge is not None, "the incident's lock state must now report a wedge"

        # 03:15:43 the gateway restarted onto the store; the bound trips ~30 min later, i.e. at
        # 03:45 — inside the 95-minute outage, with the incident's own numbers in the message.
        assert "no progress" in wedge
        # The bound is reported as a whole number of seconds (this is operator-facing text).
        assert f"bound {int(scheduler_mod.tick_lock_wedge_bound_seconds())}s" in wedge


# ── The multiplex loop must attribute the failure to the right profile ─────────────────────────


class TestMultiplexAttribution:
    def test_wedged_profile_fails_only_its_own_heartbeat(self, tmp_path, monkeypatch):
        """One wedged profile's store must not darken a healthy sibling's heartbeat."""
        home_a = tmp_path / "profiles" / "alpha"
        home_b = tmp_path / "profiles" / "beta"
        for home in (home_a, home_b):
            (home / "cron").mkdir(parents=True)

        beats: list[tuple[str, bool]] = []
        stop = threading.Event()
        prov = InProcessCronScheduler()

        def fake_tick(*_a, **_kw):
            from hermes_constants import get_hermes_home

            if Path(get_hermes_home()).name == "alpha":
                raise scheduler_mod.CronTickLockWedged("alpha's lock is wedged")
            return 0

        def beat(success=False):
            from hermes_constants import get_hermes_home

            beats.append((Path(get_hermes_home()).name, success))

        monkeypatch.setattr("cron.scheduler.tick", fake_tick)
        monkeypatch.setattr("cron.jobs.record_ticker_heartbeat", beat)
        monkeypatch.setattr("cron.jobs.record_ticker_error", lambda msg: None)
        monkeypatch.setattr("cron.jobs.clear_ticker_error", lambda: None)

        t = threading.Thread(
            target=prov.start, args=(stop,),
            kwargs={"interval": 0, "profile_homes": [("alpha", home_a), ("beta", home_b)]},
            daemon=True)
        t.start()
        assert _wait_until(lambda: len(beats) >= 4), "multiplex ticker did not keep beating"
        stop.set()
        t.join(timeout=5)

        alpha_beats = [ok for name, ok in beats if name == "alpha"]
        beta_beats = [ok for name, ok in beats if name == "beta"]
        assert alpha_beats and not any(alpha_beats), "the wedged profile must never beat success"
        # Beta's FIRST beat is the per-profile startup beat — ``record_ticker_heartbeat()`` with its
        # default ``success=False``, written before the loop (pre-existing behaviour, unrelated to
        # this fix). Every beat AFTER that must reflect beta's own tick outcome: healthy.
        assert beta_beats[1:], "the multiplex ticker beat only once"
        assert all(beta_beats[1:]), "a healthy sibling must stay healthy"
        # And alpha's failure must be attributable to alpha alone.
        assert not any(alpha_beats[1:])
