"""Escribe CIERRE_AIMS.json: manifiesto factual del estado de cada entregable."""
import datetime
import hashlib
import json
import os
import subprocess

R = r"E:\AIMS-sprint\github\AIMS-PAIEC-2026"


def sha(p):
    with open(os.path.join(R, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def size(p):
    return os.path.getsize(os.path.join(R, p))


commit = subprocess.run(["git", "-C", R, "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip()

m = {
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "overall_status": "PENDING - EXTERNALLY BLOCKED",
    "status_reason": ("No Codabench submission ID: the submission platform has been "
                      "unreachable since 2026-10-23. Not delivered, not closed."),
    "competition": {
        "name": "Predictive AI Evaluation Competition at NeurIPS 2026",
        "organiser": "Stanford AIMS",
        "rules_url": "https://aims.stanford.edu/competition",
        "deadline_utc": "2026-10-31T11:59:00Z",
        "deadline_as_published": "30 October 2026 at 23:59 Anywhere on Earth",
        "deadline_reverified_utc": "2026-10-24T09:05:00Z",
        "session": {"venue": "Sydney, Australia", "date": "2026-12-11",
                    "format": "in person, 08:00-10:15 local"},
    },
    "identity": {
        "name": "Francisco Angulo de Lafuente",
        "affiliation": "Independent researcher, Madrid, Spain",
        "email": "lareliquia.angulo@gmail.com",
        "github": "Agnuxo1",
        "huggingface": "Agnuxo",
        "team_members": ["Francisco Angulo de Lafuente"],
        "codabench_username": None,
        "codabench_username_note": ("Account exists (CodaBench acceptance email dated "
                                    "2025-10-07) but the username could not be confirmed "
                                    "because the platform is unreachable. Not invented."),
    },
    "repository": {
        "url": "https://github.com/Agnuxo1/AIMS-PAIEC-2026",
        "visibility": "public",
        "verified_logged_out": True,
        "commit": commit,
        "release_tag": "v1.0.0",
        "release_url": "https://github.com/Agnuxo1/AIMS-PAIEC-2026/releases/tag/v1.0.0",
        "licence": "MIT",
        "excluded_from_git": ["raw data", "caches", "environments", "credentials"],
    },
    "ci": {
        "last_run_id": 40081070975,
        "last_run_url": "https://github.com/Agnuxo1/AIMS-PAIEC-2026/actions/runs/40081070975",
        "conclusion": "success",
        "checks": ["synthetic metric and inference tests",
                   "submission archive validation",
                   "figure and GIF regeneration"],
        "first_run_failed": {
            "run": "37690ab",
            "cause": "scikit-learn 1.4.0 pins numpy<2; requirements were unsatisfiable from a clean install",
            "fixed_in": "e75745c"},
    },
    "artifacts": {
        "submission_original.zip": {
            "path": "submissions/submission_original.zip",
            "sha256": sha("submissions/submission_original.zip"),
            "bytes": size("submissions/submission_original.zip"),
            "passes_official_checker": True},
        "submission_context_experimental.zip": {
            "path": "submissions/submission_context_experimental.zip",
            "sha256": sha("submissions/submission_context_experimental.zip"),
            "bytes": size("submissions/submission_context_experimental.zip"),
            "passes_official_checker": True,
            "status": "experimental; returns the original predictor exactly with no peer labels"},
        "technical_report_pdf": {
            "path": "report/informe_v2.pdf",
            "sha256": sha("report/informe_v2.pdf"),
            "bytes": size("report/informe_v2.pdf"),
            "pages": 5, "template": "official NeurIPS style"},
        "technical_report_source": {"path": "report/informe_v2.tex"},
        "figures": ["docs/figures/fig1_budget_curve.png", "docs/figures/fig2_methods.png",
                    "docs/figures/fig3_cohorts.png", "docs/figures/fig4_status.png"],
        "gifs": ["docs/figures/gif1_shrinkage.gif", "docs/figures/gif2_peers.gif"],
        "gif_static_fallback": "docs/figures/gif1_shrinkage_still.png",
        "figure_generators": ["build/make_figures.py", "build/make_gifs.py"],
        "all_figures_regenerated_from": "results/*.json and results/*.csv",
    },
    "video": {
        "url": "https://youtu.be/dBfUTN5afcQ",
        "channel_id": "UCpmNBgiKW_NxkF1BBZ6v6NQ",
        "visibility": "unlisted",
        "visibility_rationale": ("The published rules do not require a video; unlisted keeps "
                                 "it reachable by link from the README."),
        "duration": "4:38", "resolution": "1920x1080", "video_codec": "h264",
        "audio": "AAC 48 kHz stereo", "bytes": 9421979,
        "narration": {"language": "English", "engine": "edge-tts",
                      "voice": "en-GB-RyanNeural", "paid_service_used": False},
        "embedded_subtitle_tracks": 0,
        "embedded_subtitle_check": "ffmpeg -i reports zero Subtitle streams",
        "burned_in_subtitles": False,
        "youtube_automatic_captions_present": True,
        "youtube_automatic_captions_removal": (
            "ATTEMPTED AND FAILED. Three routes tried in YouTube Studio (languages table row, "
            "the edit-subtitles button, the direct editor URL); the interface returned an error "
            "and did not respond to automation. Documented, not claimed as absent."),
        "video_language_metadata": ("Spanish (incorrect; the narration is English). Could not be "
                                    "changed through the same failing interface."),
        "thumbnail": "docs/video/thumbnail.png (custom, applied and verified)",
        "made_for_kids": False,
        "materials": ["docs/video/narration_script.md", "docs/video/narrate.py",
                      "docs/video/render.py", "docs/video/thumb.py",
                      "docs/video/narration.json"],
    },
    "submission": {
        "codabench_submission_ids": [],
        "accepted_submission_id": None,
        "official_scores": [],
        "final_selection": None,
        "team_registration_form_submitted": False,
        "team_registration_blocker": ("The form requires a Codabench username, which cannot be "
                                      "confirmed while the platform is down."),
    },
    "openreview": {
        "submitted": False, "submission_url": None, "submission_id": None, "receipt": None,
        "blocker": "The OpenReview form requires the selected Codabench submission ID.",
    },
    "platform_outage": {
        "host": "www.codabench.org",
        "dns_resolves_to": "129.175.8.21",
        "symptom": "TCP connect to port 443 times out",
        "attempts": 12,
        "window_utc": ["2026-10-23T21:30:00Z", "2026-10-24T09:09:00Z"],
        "independent_vantage_points": ["local network", "third-party HTTP checking service"],
        "dns_resolvers_used": ["1.1.1.1", "8.8.8.8"],
        "controls_responding_normally": ["aims.stanford.edu", "huggingface.co", "github.com"],
    },
    "organiser_contact": {
        "sent": True,
        "to": "aimslab@cs.stanford.edu",
        "from": "lareliquia.angulo@gmail.com",
        "subject": "Codabench unreachable - Predictive AI Evaluation Competition submission route",
        "sent_local_time": "2026-10-24 11:12 Europe/Madrid",
        "verified_in_sent_folder": True,
        "prior_thread_existed": False,
        "asks": ["is the outage known and expected recovery",
                 "alternative written route preserving eligibility",
                 "how to handle the submission ID field",
                 "what to prepare now"],
        "reply_received": False,
        "note": ("Sending this email is not an acceptance and does not substitute the normal "
                 "submission flow."),
    },
    "results_claimed": {
        "protocol": ("independent re-evaluation: 151 benchmarks, 13,200 eligible pairs, "
                     "105,600 scenarios, 142 components, 8 seeds, 5 outer folds"),
        "empirical_mean_reference": 0.236814,
        "submitted_predictor": 0.205984,
        "difference": -0.030830,
        "ci95": [-0.038426, -0.027214],
        "benchmarks_improved": "133/151",
        "nature": ("LOCAL VALIDATION on public benchmarks held out from each other. Not an "
                   "official score. The reference is deliberately simple."),
        "conditional_extension": {
            "difference": -0.002562,
            "ci99_five_candidates": [-0.006257, -0.000551],
            "environment": "simulated 4-subject cohort",
            "status": "EXPERIMENTAL; real peer-label availability on the platform is UNVERIFIED",
            "caveat": ("Estimates the nested selection procedure, not an untouched test of the "
                       "final parameter. Identical to the original when no peer labels are supplied.")},
        "ranking_or_acceptance_claimed": False,
    },
    "data_provenance": {
        "dataset": "aims-foundations/measurement-db",
        "pinned_revision": "54074453da7b9911be0f48e4c560be547bc84d21",
        "files_pinned": 890,
        "bytes_required": 5354371887,
        "downloader": "audit/download_pinned.py",
    },
    "security": {
        "credentials_in_public_artifacts": False,
        "credentials_file_read_in_this_task": False,
        "pending_owner_action": ("Key rotation for the previously exposed credentials file remains "
                                 "the owner's decision; nothing was revoked."),
    },
    "next_steps": [
        "Retry https://www.codabench.org/competitions/17828/ at intervals until 2026-10-30",
        "On access: confirm the Codabench username, complete the team registration form, upload "
        "submission_original.zip, record the submission ID and its state",
        "If quota and rules allow, also test the contextual archive and check whether peer labels "
        "are genuinely supplied, without identifying anonymous subjects or benchmarks",
        "Select the final entry on the available evidence and document the decision",
        "Submit the report through OpenReview with the selected submission ID and the public links",
        "Follow any alternative route the organisers provide and record the receipt",
    ],
    "future_obligations": [
        "In-person session in Sydney on 2026-12-11, 08:00-10:15 local, if selected for oral or "
        "poster. No travel support published.",
        "Consideration for oral or poster requires the technical report and the release of the "
        "code used; the code is already public.",
        "Organisers rerun the released method to check reproducibility.",
    ],
}

out = os.path.join(R, "CIERRE_AIMS.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(m, fh, ensure_ascii=False, indent=1)
print("CIERRE_AIMS.json escrito:", os.path.getsize(out), "bytes")
