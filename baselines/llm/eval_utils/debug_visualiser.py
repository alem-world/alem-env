"""Generate a self-contained HTML visualisation from a debug JSONL file."""

import html
import json
import os


def generate_debug_html(debug_jsonl_path, episode_json_path=None, output_path=None):
    """Read a _debug.jsonl and optional episode .json, write an interactive HTML file.

    Args:
        debug_jsonl_path: Path to the _debug.jsonl file.
        episode_json_path: Optional path to the episode .json (summary stats).
        output_path: Where to write the HTML. Defaults to same dir as the JSONL
                     with _debug.html suffix.
    """
    if output_path is None:
        output_path = debug_jsonl_path.replace("_debug.jsonl", "_debug.html")

    # Load steps
    steps = []
    with open(debug_jsonl_path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError as e:
                    import logging

                    logging.warning(
                        f"Skipping malformed JSONL line {lineno} in {debug_jsonl_path}: {e}"
                    )

    # Load episode summary if available
    episode_summary = None
    if episode_json_path and os.path.exists(episode_json_path):
        with open(episode_json_path, encoding="utf-8") as f:
            episode_summary = json.load(f)

    steps_json = json.dumps(steps)
    summary_json = json.dumps(episode_summary) if episode_summary else "null"

    title = os.path.basename(debug_jsonl_path).replace("_debug.jsonl", "")

    html_content = _build_html(title, steps_json, summary_json)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


def generate_step_log_txt(debug_jsonl_path, output_path=None):
    """Write a human-readable text file showing messages and reasoning per step.

    For each step shows: the new observation, recent prompt history, reasoning,
    and extracted action for every agent.

    Args:
        debug_jsonl_path: Path to the _debug.jsonl file.
        output_path: Where to write the text file. Defaults to _steps.txt.
    """
    if output_path is None:
        output_path = debug_jsonl_path.replace("_debug.jsonl", "_steps.txt")

    steps = []
    with open(debug_jsonl_path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError as e:
                    import logging

                    logging.warning(
                        f"Skipping malformed JSONL line {lineno} in {debug_jsonl_path}: {e}"
                    )

    SEP = "─" * 80

    # Separate debrief record from step records
    debrief_record = None
    game_steps = []
    for s in steps:
        if s.get("type") == "debrief":
            debrief_record = s
        else:
            game_steps.append(s)

    with open(output_path, "w", encoding="utf-8") as out:
        for step_data in game_steps:
            step_num = step_data.get("step", "?")
            rewards = step_data.get("rewards", [])
            out.write(f"\n{'━' * 80}\n")
            out.write(f"  STEP {step_num}   rewards={rewards}\n")
            out.write(f"{'━' * 80}\n")

            agents = step_data.get("agents", {})
            for ag_idx, ag in sorted(agents.items()):
                out.write(f"\n  ── Agent {ag_idx} ──\n")

                # --- Prompt messages (skip system/instruction, show recent history) ---
                messages = ag.get("prompt_messages", [])
                if messages:
                    # Show last N messages before the final user message
                    # (skip the first user message which is the system instruction)
                    non_system = [m for m in messages[1:]]  # drop instruction
                    # The last message is the current observation; show all others as history
                    history = non_system[:-1]
                    current_obs_msg = non_system[-1] if non_system else None

                    if history:
                        out.write("  [HISTORY]\n")
                        for msg in history:
                            role = msg.get("role", "?").upper()
                            content = msg.get("content", "").strip()
                            out.write(f"    {role}: {content}\n")

                    if current_obs_msg:
                        out.write("  [CURRENT OBS]\n")
                        out.write(f"    {current_obs_msg.get('content', '').strip()}\n")
                else:
                    # Fallback: show obs from snapshot
                    obs = (ag.get("obs_long_term") or "").strip()
                    if obs:
                        out.write("  [OBS]\n")
                        out.write(f"    {obs}\n")

                # --- Raw LLM output (full completion before extraction) ---
                raw_completion = ag.get("llm_raw_output")
                if raw_completion is None:
                    raw_completion = ag.get("raw_completion")
                raw_completion = (raw_completion or "").strip()
                if raw_completion:
                    out.write("  [RAW LLM OUTPUT]\n")
                    for line in raw_completion.split("\n"):
                        out.write(f"    {line}\n")
                elif ag.get("llm_raw_output", None) == "":
                    out.write("  [RAW LLM OUTPUT]\n")
                    out.write("    (empty)\n")

                # --- Raw reasoning (vLLM thinking tokens, if any) ---
                raw_reasoning = (ag.get("raw_reasoning") or "").strip()
                if raw_reasoning:
                    out.write("  [RAW REASONING (thinking tokens)]\n")
                    for line in raw_reasoning.split("\n"):
                        out.write(f"    {line}\n")

                # --- Cleaned reasoning ---
                reasoning = (ag.get("reasoning") or "").strip()
                out.write("  [REASONING]\n")
                out.write(f"    {reasoning if reasoning else '(none)'}\n")

                # --- Action ---
                action = ag.get("parsed_action") or ag.get("action") or "?"
                out.write(f"  [ACTION] {action}\n")

                # --- Communication ---
                sent_msg = ag.get("message_sent")
                recv_msgs = ag.get("messages_received", [])
                if sent_msg or recv_msgs:
                    out.write("  [COMMUNICATION]\n")
                    if recv_msgs:
                        out.write("    Received:\n")
                        for m in recv_msgs:
                            sender = m.get("sender_idx", "?")
                            role = f" ({m['sender_role']})" if m.get("sender_role") else ""
                            out.write(f"      Agent {sender}{role}: {m.get('content', '')}\n")
                    if sent_msg and sent_msg.get("content"):
                        out.write(f"    Sent: {sent_msg['content']}\n")

                # --- Scratchpad ---
                scratchpad = ag.get("scratchpad")
                if scratchpad is not None:
                    out.write("  [SCRATCHPAD]\n")
                    for line in str(scratchpad).split("\n"):
                        out.write(f"    {line}\n")

        # --- Post-episode debriefs ---
        if debrief_record and debrief_record.get("agents"):
            out.write(f"\n{'━' * 80}\n")
            out.write("  POST-EPISODE DEBRIEF\n")
            out.write(f"{'━' * 80}\n")
            for key, text in sorted(debrief_record["agents"].items()):
                out.write(f"\n  ── {key} ──\n")
                for line in (text or "(no debrief)").split("\n"):
                    out.write(f"    {line}\n")

    return output_path


# ---------------------------------------------------------------------------
# The JS is kept as a plain string constant (no f-string) to avoid any
# brace / quote escaping issues.  The two data blobs (STEPS, SUMMARY)
# are spliced in via simple string concatenation.
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #e6edf3; --text-dim: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --red: #f85149; --yellow: #d29922; --orange: #db6d28;
  --agent0: #58a6ff; --agent1: #3fb950; --agent2: #d29922; --agent3: #f778ba;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: "SF Mono","Fira Code","Cascadia Code",monospace; background: var(--bg); color: var(--text); font-size: 13px; }

.header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; }
.header h1 { font-size: 15px; font-weight: 600; }
.header .meta { color: var(--text-dim); font-size: 12px; }

.controls { background: var(--surface); border-bottom: 1px solid var(--border); padding: 8px 20px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; position: sticky; top: 45px; z-index: 99; }
.controls button { background: var(--border); color: var(--text); border: none; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 12px; }
.controls button:hover { background: var(--accent); color: #000; }
.controls button.active { background: var(--accent); color: #000; }
.controls input[type=range] { flex: 1; max-width: 400px; accent-color: var(--accent); }
.controls .step-label { color: var(--accent); font-weight: 600; min-width: 90px; }

.summary-bar { background: var(--surface); border-bottom: 1px solid var(--border); padding: 8px 20px; display: flex; gap: 24px; flex-wrap: wrap; }
.summary-bar .stat { display: flex; flex-direction: column; gap: 2px; }
.summary-bar .stat-label { color: var(--text-dim); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
.summary-bar .stat-value { font-size: 14px; font-weight: 600; }

.main { padding: 16px 20px; }

.step-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 12px; }
.step-header { padding: 10px 16px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
.step-header .step-num { font-weight: 700; color: var(--accent); }
.step-header .rewards { font-size: 12px; color: var(--text-dim); }
.reward-pos { color: var(--green); }
.reward-neg { color: var(--red); }
.reward-zero { color: var(--text-dim); }

.agents-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); }
.agent-col { padding: 12px 16px; border-right: 1px solid var(--border); }
.agent-col:last-child { border-right: none; }
.agent-label { font-weight: 700; font-size: 12px; margin-bottom: 8px; padding: 2px 8px; border-radius: 3px; display: inline-block; }
.agent-label-0 { background: rgba(88,166,255,0.15); color: var(--agent0); }
.agent-label-1 { background: rgba(63,185,80,0.15); color: var(--agent1); }
.agent-label-2 { background: rgba(210,153,34,0.15); color: var(--agent2); }
.agent-label-3 { background: rgba(247,120,186,0.15); color: var(--agent3); }

.field { margin-bottom: 8px; }
.field-label { color: var(--text-dim); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 3px; }

.action-row { display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; margin-bottom: 6px; }
.raw-output { background: #1c2128; padding: 6px 10px; border-radius: 4px; border-left: 3px solid var(--orange); }
.parsed-action { background: #1c2128; padding: 6px 10px; border-radius: 4px; border-left: 3px solid var(--green); font-weight: 600; }
.action-mismatch { border-left-color: var(--red); }
.reasoning-block { background: #1c2128; padding: 6px 10px; border-radius: 4px; border-left: 3px solid var(--accent); color: var(--text-dim); font-style: italic; }

.msg-sent-block { background: #1c2128; padding: 6px 10px; border-radius: 4px; border-left: 3px solid var(--yellow); font-style: italic; }
.msg-recv-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 4px; }
.msg-recv-item { background: #1c2128; padding: 5px 10px; border-radius: 4px; border-left: 3px solid var(--text-dim); }
.msg-recv-item-0 { border-left-color: var(--agent0); }
.msg-recv-item-1 { border-left-color: var(--agent1); }
.msg-recv-item-2 { border-left-color: var(--agent2); }
.msg-recv-item-3 { border-left-color: var(--agent3); }
.msg-sender-tag { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
.msg-sender-tag-0 { color: var(--agent0); }
.msg-sender-tag-1 { color: var(--agent1); }
.msg-sender-tag-2 { color: var(--agent2); }
.msg-sender-tag-3 { color: var(--agent3); }
.msg-content { color: var(--text); }

.scratchpad-block { background: #1c2128; padding: 6px 10px; border-radius: 4px; border-left: 3px solid #a371f7; white-space: pre-wrap; font-size: 12px; }

.obs-block { background: #1c2128; padding: 8px 10px; border-radius: 4px; max-height: 200px; overflow-y: auto; font-size: 12px; line-height: 1.4; white-space: pre-wrap; word-break: break-word; }
.obs-block.expanded { max-height: none; }
.expand-btn { color: var(--accent); cursor: pointer; font-size: 11px; border: none; background: none; font-family: inherit; padding: 2px 0; }

.agent-image { margin: 6px 0; }
.agent-image img { max-width: 100%; border-radius: 4px; border: 1px solid var(--border); image-rendering: pixelated; }

.prompt-block { background: #1c2128; border-radius: 4px; max-height: 300px; overflow-y: auto; font-size: 12px; line-height: 1.4; }
.prompt-block.expanded { max-height: none; }
.prompt-msg { padding: 6px 10px; border-bottom: 1px solid var(--border); }
.prompt-msg:last-child { border-bottom: none; }
.prompt-msg-role { font-weight: 700; font-size: 10px; text-transform: uppercase; margin-bottom: 2px; }
.prompt-msg-role-user { color: var(--accent); }
.prompt-msg-role-assistant { color: var(--green); }
.prompt-msg-role-system { color: var(--yellow); }
.prompt-msg-content { white-space: pre-wrap; word-break: break-word; }

.filter-row { display: flex; gap: 8px; align-items: center; }
.filter-row label { color: var(--text-dim); font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 4px; }

.debrief-panel { background: var(--surface); border-top: 2px solid var(--accent); padding: 16px 20px; margin-top: 8px; }
.debrief-panel h2 { font-size: 13px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px; }
.debrief-agents { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 12px; }
.debrief-agent { background: #1c2128; border-radius: 6px; padding: 10px 14px; border-top: 3px solid var(--text-dim); }
.debrief-agent-0 { border-top-color: var(--agent0); }
.debrief-agent-1 { border-top-color: var(--agent1); }
.debrief-agent-2 { border-top-color: var(--agent2); }
.debrief-agent-3 { border-top-color: var(--agent3); }
.debrief-agent-label { font-weight: 700; font-size: 11px; text-transform: uppercase; margin-bottom: 8px; }
.debrief-agent-label-0 { color: var(--agent0); }
.debrief-agent-label-1 { color: var(--agent1); }
.debrief-agent-label-2 { color: var(--agent2); }
.debrief-agent-label-3 { color: var(--agent3); }
.debrief-text { white-space: pre-wrap; font-size: 12px; line-height: 1.55; color: var(--text); }
"""

_JS = """\
// Filter out non-step records (e.g. debrief summary appended at episode end)
var _debriefRecord = null;
STEPS = STEPS.filter(function(s) {
  if (s.type === "debrief") { _debriefRecord = s; return false; }
  return true;
});

var mode = "all";
var currentStep = 0;

var _numAgents = Object.keys(STEPS[0] && STEPS[0].agents || {}).length;
document.getElementById("metaInfo").textContent =
  STEPS.length + " steps, " + _numAgents + " agents" +
  (SUMMARY ? ", mean return: " + (SUMMARY.episode_return).toFixed(2) : "");

document.getElementById("stepSlider").max = STEPS.length - 1;

if (SUMMARY) {
  var bar = document.getElementById("summaryBar");
  var numAgents = SUMMARY.num_agents || Object.keys(STEPS[0] && STEPS[0].agents || {}).length;
  var meanReturn = (SUMMARY.episode_return || 0).toFixed(2);
  var stats = [
    ["Steps", SUMMARY.num_steps],
    ["Mean Episode Return", meanReturn],
    ["Input Tokens", (SUMMARY.input_tokens || 0).toLocaleString()],
    ["Output Tokens", (SUMMARY.output_tokens || 0).toLocaleString()],
    ["Action Parse", ((SUMMARY.action_parse_rate || 0) * 100).toFixed(1) + "%"],
  ];
  if (SUMMARY.comm_parse_rate != null) {
    stats.push(["Comm Parse", ((SUMMARY.comm_parse_rate || 0) * 100).toFixed(1) + "% (" + (SUMMARY.comm_parsed || 0) + "/" + (SUMMARY.comm_attempted || 0) + ")"]);
  }
  if (SUMMARY.scratchpad_parse_rate != null) {
    stats.push(["Scratchpad Parse", ((SUMMARY.scratchpad_parse_rate || 0) * 100).toFixed(1) + "% (" + (SUMMARY.scratchpad_parsed || 0) + "/" + (SUMMARY.scratchpad_attempted || 0) + ")"]);
  }
  // Achievement stats from user_info
  var ui = SUMMARY.user_info || {};
  // Count total normal vs coordination achievements from agent_0's achievement dict
  var totalNormal = 0, totalCoord = 0;
  var a0 = SUMMARY.agent_0 || {};
  var a0Ach = a0.achievements || {};
  Object.keys(a0Ach).forEach(function(k) {
    if (k.indexOf("COORD_") === 0 || k === "HANDOVER_COMPLETE") {
      totalCoord++;
    } else {
      totalNormal++;
    }
  });
  var normalAch = ui["Team/normal_achievements"] || 0;
  var coordAch = ui["Team/coordination_achievements"] || 0;
  var normalPct = totalNormal > 0 ? ((normalAch / totalNormal) * 100).toFixed(1) : "0.0";
  var coordPct = totalCoord > 0 ? ((coordAch / totalCoord) * 100).toFixed(1) : "0.0";
  stats.push(["Normal Achievements", normalPct + "% (" + normalAch + "/" + totalNormal + ")"]);
  stats.push(["Coord Achievements", coordPct + "% (" + coordAch + "/" + totalCoord + ")"]);
  for (var i = 0; i < numAgents; i++) {
    stats.push(["Agent " + i + " Return", (SUMMARY["agent_" + i + "_return"] || 0).toFixed(2)]);
  }
  bar.innerHTML = stats.map(function(s) {
    return '<div class="stat"><span class="stat-label">' + s[0] + '</span><span class="stat-value">' + s[1] + '</span></div>';
  }).join("");
}

function toggleExpand(btn) {
  var block = btn.parentElement.nextElementSibling;
  block.classList.toggle("expanded");
  btn.textContent = block.classList.contains("expanded") ? "collapse" : "expand";
}

function escHtml(s) {
  if (s == null) return '<span style="color:var(--text-dim)">null</span>';
  var div = document.createElement("div");
  div.textContent = String(s);
  return div.innerHTML;
}

function rewardClass(r) {
  if (r > 0) return "reward-pos";
  if (r < 0) return "reward-neg";
  return "reward-zero";
}

function renderPromptMessages(msgs) {
  if (!msgs || !msgs.length) return "";
  return msgs.map(function(m) {
    var roleClass = "prompt-msg-role prompt-msg-role-" + m.role;
    return '<div class="prompt-msg"><div class="' + roleClass + '">' + m.role + '</div><div class="prompt-msg-content">' + escHtml(m.content) + '</div></div>';
  }).join("");
}

function renderStep(step, idx) {
  var agents = step.agents || {};
  var agentIds = Object.keys(agents).sort();
  var rewards = step.rewards || [];

  var rewardsHtml = rewards.map(function(r, i) {
    return '<span class="' + rewardClass(r) + '">A' + i + ":" + r.toFixed(2) + "</span>";
  }).join(" ");

  var agentsHtml = agentIds.map(function(aid) {
    var a = agents[aid];
    var mismatch = a.llm_raw_output !== a.parsed_action;
    var labelClass = "agent-label agent-label-" + aid;
    var s = "";

    // Image
    if (a.image_base64) {
      s += '<div class="field image-field"><div class="field-label">Pixel Observation</div>';
      s += '<div class="agent-image"><img src="data:image/png;base64,' + a.image_base64 + '" alt="Agent ' + aid + ' view"></div></div>';
    }

    // Action row
    s += '<div class="field"><div class="field-label">Action</div>';
    s += '<div class="action-row">';
    var rawOutput = a.llm_raw_output;
    if (rawOutput === "") rawOutput = "(empty)";
    s += '<div class="raw-output' + (mismatch ? " action-mismatch" : "") + '"><span class="field-label" style="margin:0">LLM raw &rarr;</span> ' + escHtml(rawOutput) + "</div>";
    s += '<div class="parsed-action"><span class="field-label" style="margin:0">Parsed &rarr;</span> ' + escHtml(a.parsed_action) + "</div>";
    s += "</div></div>";

    // Reasoning
    if (a.reasoning) {
      s += '<div class="field reasoning-field"><div class="field-label">Reasoning</div>';
      s += '<div class="reasoning-block">' + escHtml(a.reasoning) + "</div></div>";
    }

    // Communication (sent + received) — separate section
    var recvMsgs = a.messages_received || [];
    var sentMsg = a.message_sent;
    if (recvMsgs.length || sentMsg) {
      s += '<div class="field msg-field">';
      s += '<div class="field-label">Communication</div>';
      if (sentMsg && sentMsg.content) {
        s += '<div class="field-label" style="font-size:9px;margin-top:4px">Sent</div>';
        s += '<div class="msg-sent-block">' + escHtml(sentMsg.content) + '</div>';
      }
      if (recvMsgs.length) {
        s += '<div class="field-label" style="font-size:9px;margin-top:6px">Received (' + recvMsgs.length + ')</div>';
        s += '<ul class="msg-recv-list">';
        recvMsgs.forEach(function(m) {
          var sidx = m.sender_idx != null ? String(m.sender_idx) : "?";
          var role = m.sender_role ? " (" + m.sender_role + ")" : "";
          var itemClass = "msg-recv-item msg-recv-item-" + sidx;
          var tagClass = "msg-sender-tag msg-sender-tag-" + sidx;
          s += '<li class="' + itemClass + '">';
          s += '<div class="' + tagClass + '">Agent ' + sidx + role + '</div>';
          s += '<div class="msg-content">' + escHtml(m.content) + '</div>';
          s += '</li>';
        });
        s += '</ul>';
      }
      s += '</div>';
    }

    // Scratchpad (memory) — separate section
    if (a.scratchpad !== null && a.scratchpad !== undefined) {
      s += '<div class="field scratchpad-field"><div class="field-label">Scratchpad (Memory)</div>';
      s += '<div class="scratchpad-block">' + escHtml(a.scratchpad || "(empty)") + '</div></div>';
    }

    // Observations
    s += '<div class="field obs-field"><div class="field-label">Observation (long-term) <button class="expand-btn" onclick="toggleExpand(this)">expand</button></div>';
    s += '<div class="obs-block">' + escHtml(a.obs_long_term) + "</div></div>";

    s += '<div class="field obs-field"><div class="field-label">Status (short-term) <button class="expand-btn" onclick="toggleExpand(this)">expand</button></div>';
    s += '<div class="obs-block">' + escHtml(a.obs_short_term) + "</div></div>";

    // Full prompt history
    if (a.prompt_messages && a.prompt_messages.length) {
      s += '<div class="field prompt-field"><div class="field-label">Full LLM Prompt (' + a.prompt_messages.length + ' messages) <button class="expand-btn" onclick="toggleExpand(this)">expand</button></div>';
      s += '<div class="prompt-block">' + renderPromptMessages(a.prompt_messages) + "</div></div>";
    }

    return '<div class="agent-col"><span class="' + labelClass + '">Agent ' + aid + "</span>" + s + "</div>";
  }).join("");

  return '<div class="step-card" id="step-' + idx + '">' +
    '<div class="step-header"><span class="step-num">Step ' + step.step + '</span><span class="rewards">' + rewardsHtml + "</span></div>" +
    '<div class="agents-grid">' + agentsHtml + "</div></div>";
}

function renderAll() {
  var container = document.getElementById("stepsContainer");
  if (mode === "all") {
    container.innerHTML = STEPS.map(function(s, i) { return renderStep(s, i); }).join("");
  } else {
    container.innerHTML = renderStep(STEPS[currentStep], currentStep);
  }
  applyFilters();
}

function applyFilters() {
  var showObs = document.getElementById("showObs").checked;
  var showReasoning = document.getElementById("showReasoning").checked;
  var showImages = document.getElementById("showImages").checked;
  var showPrompt = document.getElementById("showPrompt").checked;
  var showMsgs = document.getElementById("showMsgs").checked;
  var showScratchpad = document.getElementById("showScratchpad").checked;
  document.querySelectorAll(".obs-field").forEach(function(el) { el.style.display = showObs ? "" : "none"; });
  document.querySelectorAll(".reasoning-field").forEach(function(el) { el.style.display = showReasoning ? "" : "none"; });
  document.querySelectorAll(".image-field").forEach(function(el) { el.style.display = showImages ? "" : "none"; });
  document.querySelectorAll(".prompt-field").forEach(function(el) { el.style.display = showPrompt ? "" : "none"; });
  document.querySelectorAll(".msg-field").forEach(function(el) { el.style.display = showMsgs ? "" : "none"; });
  document.querySelectorAll(".scratchpad-field").forEach(function(el) { el.style.display = showScratchpad ? "" : "none"; });
}

function goToStep(idx) {
  currentStep = Math.max(0, Math.min(STEPS.length - 1, idx));
  document.getElementById("stepSlider").value = currentStep;
  document.getElementById("stepLabel").textContent = "Step " + STEPS[currentStep].step;
  if (mode === "single") renderAll();
  else {
    var el = document.getElementById("step-" + currentStep);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

document.getElementById("btnAll").addEventListener("click", function() {
  mode = "all";
  document.getElementById("btnAll").classList.add("active");
  document.getElementById("btnSingle").classList.remove("active");
  renderAll();
});
document.getElementById("btnSingle").addEventListener("click", function() {
  mode = "single";
  document.getElementById("btnSingle").classList.add("active");
  document.getElementById("btnAll").classList.remove("active");
  renderAll();
});
document.getElementById("stepSlider").addEventListener("input", function(e) { goToStep(parseInt(e.target.value)); });
document.getElementById("btnPrev").addEventListener("click", function() { goToStep(currentStep - 1); });
document.getElementById("btnNext").addEventListener("click", function() { goToStep(currentStep + 1); });
document.getElementById("showObs").addEventListener("change", applyFilters);
document.getElementById("showReasoning").addEventListener("change", applyFilters);
document.getElementById("showImages").addEventListener("change", applyFilters);
document.getElementById("showPrompt").addEventListener("change", applyFilters);
document.getElementById("showMsgs").addEventListener("change", applyFilters);
document.getElementById("showScratchpad").addEventListener("change", applyFilters);

document.addEventListener("keydown", function(e) {
  if (e.key === "ArrowLeft") { e.preventDefault(); goToStep(currentStep - 1); }
  if (e.key === "ArrowRight") { e.preventDefault(); goToStep(currentStep + 1); }
});

renderAll();

// Render debrief panel — prefers SUMMARY.debriefs, falls back to _debriefRecord
(function() {
  var debriefs = (SUMMARY && SUMMARY.debriefs) || (_debriefRecord && _debriefRecord.agents) || null;
  if (!debriefs) return;
  var panel = document.getElementById("debriefPanel");
  var container = document.getElementById("debriefAgents");
  var agentKeys = Object.keys(debriefs).sort();
  if (!agentKeys.length) return;
  panel.style.display = "";
  container.innerHTML = agentKeys.map(function(key) {
    var idx = key.replace("agent_", "");
    var text = debriefs[key] || "(no debrief)";
    return (
      '<div class="debrief-agent debrief-agent-' + idx + '">' +
      '<div class="debrief-agent-label debrief-agent-label-' + idx + '">Agent ' + idx + '</div>' +
      '<div class="debrief-text">' + escHtml(text) + '</div>' +
      '</div>'
    );
  }).join("");
})();
"""

_BODY_HTML = """\
<div class="header">
  <h1>Debug Viewer</h1>
  <span class="meta" id="metaInfo"></span>
</div>

<div class="summary-bar" id="summaryBar"></div>

<div class="controls">
  <button id="btnPrev" title="Previous step">&larr;</button>
  <button id="btnNext" title="Next step">&rarr;</button>
  <input type="range" id="stepSlider" min="0" value="0">
  <span class="step-label" id="stepLabel">Step 0</span>
  <span style="color:var(--text-dim)">|</span>
  <button id="btnAll" class="active">All Steps</button>
  <button id="btnSingle">Single Step</button>
  <span style="color:var(--text-dim)">|</span>
  <div class="filter-row">
    <label><input type="checkbox" id="showObs" checked> Obs</label>
    <label><input type="checkbox" id="showReasoning" checked> Reasoning</label>
    <label><input type="checkbox" id="showImages" checked> Images</label>
    <label><input type="checkbox" id="showMsgs" checked> Messages</label>
    <label><input type="checkbox" id="showScratchpad" checked> Scratchpad</label>
    <label><input type="checkbox" id="showPrompt"> Full Prompt</label>
  </div>
</div>

<div class="main" id="stepsContainer"></div>
<div class="debrief-panel" id="debriefPanel" style="display:none">
  <h2>Post-Episode Debrief</h2>
  <div class="debrief-agents" id="debriefAgents"></div>
</div>
"""


def _build_html(title, steps_json, summary_json):
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>Debug: " + html.escape(title) + "</title>\n"
        "<style>\n" + _CSS + "</style>\n"
        "</head>\n"
        "<body>\n" + _BODY_HTML + "<script>\n"
        "var STEPS = " + steps_json + ";\n"
        "var SUMMARY = " + summary_json + ";\n" + _JS + "</script>\n"
        "</body>\n"
        "</html>"
    )
