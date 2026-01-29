import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import Levenshtein as LEV
import matplotlib.pyplot as PLT
import plotly.express as PX
import plotly.graph_objects as GO
import plotly.io as pio
import s3fs
import wandb
from upsetplot import UpSet, from_memberships

from prog_repair_bench.data_preprocess.s3_folders_manager import is_s3_path
from prog_repair_bench.processors.process_results import read_jsonl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plotly Template Configuration
# ---------------------------------------------------------------------------

palette = PX.colors.qualitative.Bold
custom_template = GO.layout.Template(layout=GO.Layout(colorway=palette))
pio.templates["bench_style"] = custom_template
pio.templates.default = "bench_style"


# ===========================================================================
# DATA EXTRACTION UTILITIES
# ===========================================================================


def _extract_task_id(item: Dict[str, Any]) -> Optional[int]:
    """Extract the benchmark task ID from a result item."""
    return item.get("dp_item", {}).get("idx")


def _extract_ground_truth_code(item: Dict[str, Any]) -> str:
    """Extract concatenated declaration + body from ground truth method."""
    method = item.get("dp_item", {}).get("method", {})
    declaration = method.get("declaration", "")
    body = method.get("body", "")
    return f"{declaration}\n{body}".strip()


def _extract_prompt(item: Dict[str, Any], turn_idx: int = 0) -> str:
    """Extract the model prompt for a given turn index."""
    messages = item.get("messages", [])

    idx = 2 * turn_idx + 1 if turn_idx > 1 else 1
    if idx >= len(messages):
        return ""

    msg = messages[idx]
    if isinstance(msg, dict):
        return str(msg.get("content", "")).strip()
    return getattr(msg, "content", "").strip()


def _extract_generated_code(item: Dict[str, Any], turn_idx: int = -1) -> str:
    """Return generated code for a given turn index (default: last)."""
    codes = item.get("generate_codes")
    if -len(codes) <= turn_idx < len(codes):
        return str(codes[turn_idx])
    return ""


def _extract_raw_response(item: Dict[str, Any], turn_idx: int = -1) -> str:
    """Return raw model response for a given turn index (default: last)."""
    responses = item.get("responses_raw", [])
    if not responses:
        return ""

    if -len(responses) <= turn_idx < len(responses):
        response = responses[turn_idx]
        if isinstance(response, dict):
            return str(response.get("content", ""))
        elif hasattr(response, "content"):
            return str(response.content)
        else:
            return str(response)
    return ""


def _extract_error_type(item: dict, short: bool = True, turn_idx: Optional[int] = -1) -> str:
    """
    Extract error type (or full stacktrace if short=False) for a specific turn.

    Returns:
        - "OK" for successful status
        - "NOT_RUN" for not run status
        - Error type string for failures
        - Empty string when no stacktrace/diagnostics available
    """
    test_results = item.get("test_results", [])
    if not test_results:
        return ""

    idx = turn_idx if 0 <= turn_idx < len(test_results) else -1
    res = test_results[idx]
    status = res.get("status")
    details = res.get("details", None)

    if status == "OK":
        return "OK"

    if not details or status == "NOT_RUN":
        return "NOT_RUN" if short and status == "NOT_RUN" else res.get("test_output", "")

    stacktrace = details[-1].get("stacktrace", "")
    if not short or not stacktrace:
        return stacktrace.strip()

    traceback_idx = stacktrace.find("Traceback")
    if traceback_idx != -1:
        stacktrace = stacktrace[traceback_idx:]

    error_words = [
        word for line in stacktrace.splitlines() for word in line.split() if "Error" in word
    ]

    if error_words:
        return error_words[-1].split(":")[0][:15].strip()

    lines = stacktrace.splitlines()
    if lines:
        return lines[-1].split(":", 1)[0].strip()

    return ""


# ===========================================================================
# DATA FILTERING UTILITIES
# ===========================================================================


def _filter_task_ids_by_status(res: List[dict], status: str) -> Set[int]:
    """Return a set of task IDs for which the test status matches the given status."""
    return {
        tid
        for item in res
        if item.get("test_status") == status and (tid := _extract_task_id(item)) is not None
    }


# ===========================================================================
# FILE I/O UTILITIES
# ===========================================================================


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL file from local disk or S3."""
    if is_s3_path(path):
        data: List[Dict[str, Any]] = []
        fs = s3fs.S3FileSystem()
        with fs.open(str(path), "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    else:
        return read_jsonl(path)


def _load_competitors_once(cfg: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Load competitor result files once."""
    loaded = []
    logger.info(f"Loading competitors from config with {len(cfg)} entries")

    for comp in cfg:
        path = comp.get("path")
        if not path:
            logger.warning("  Competitor: No path provided, skipping")
            continue

        label = comp.get("label", "")
        logger.info(f"  Loading competitor '{label}' from {path}")

        try:
            res = _load_jsonl(str(path))
            loaded.append((label, res))
            logger.info(f"    ✓ Loaded {len(res)} results for {label}")
        except Exception as e:
            logger.error(f"    ✗ Failed to load {label} from {path}: {e}", exc_info=True)
            print(f"Warning: failed to load {label} from {path}: {e}")

    logger.info(f"Successfully loaded {len(loaded)} out of {len(cfg)} competitors")
    return loaded


# ===========================================================================
# LEVENSHTEIN DISTANCE UTILITIES
# ===========================================================================


def _collect_lev_rows_by_turn(
        res: List[Dict[str, Any]],
) -> Dict[Union[int, str], Dict[str, List[int]]]:
    """
    Aggregate Levenshtein distances per turn and overall.

    Returns:
        Mapping: {turn_index (1-based) -> {status -> [distances]}, 'ALL' -> aggregate}
    """
    per_turn: Dict[int, Dict[str, List[int]]] = {}
    overall: Dict[str, List[int]] = {}

    for item in res:
        gt = _extract_ground_truth_code(item)
        turns = item.get("generate_codes", []) or []
        test_results = item.get("test_results", []) or []

        if not gt or not turns:
            continue

        for turn_idx, gen in enumerate(turns):
            distance = int(LEV.distance(gt, str(gen)))
            status = "UNKNOWN"

            if 0 <= turn_idx < len(test_results):
                status = str(test_results[turn_idx].get("status", "UNKNOWN"))

            turn_key = turn_idx + 1
            per_turn.setdefault(turn_key, {}).setdefault(status, []).append(distance)
            overall.setdefault(status, []).append(distance)

    combined: Dict[Union[int, str], Dict[str, List[int]]] = dict(per_turn)
    if overall:
        combined["ALL"] = overall

    return combined


# ===========================================================================
# COLOR MAPPING UTILITIES
# ===========================================================================


def _build_specific_model_colors(labels: List[str]) -> Dict[str, str]:
    """Assign fixed colors to first three models."""
    fixed = ["#d62728", "#1f77b4", "#bcbd22"]  # red, blue, yellow
    return {lbl: fixed[i] for i, lbl in enumerate(labels[:3])}


def _intersection_color_for(combo: Tuple[str, ...], labels: List[str]) -> str:
    """Return fixed intersection colors based on first three models, else gray."""
    idxs = {labels.index(l) for l in combo if l in labels}

    if len(idxs) == 3 and {0, 1, 2}.issubset(idxs):
        return "#000000"  # overall intersection → black
    if idxs == {0, 1}:
        return "#9467bd"  # purple (1&2)
    if idxs == {0, 2}:
        return "#ff7f0e"  # orange (1&3)
    if idxs == {1, 2}:
        return "#2ca02c"  # green (2&3)

    return "#7f7f7f"  # gray


def _status_to_color(status: str) -> str:
    """Map test status to color."""
    mapping = {
        "OK": "#2ca02c",  # green
        "FAILED": "#d62728",  # red
        "NOT_RUN": "#1f77b4",  # blue
        "SOLVED": "#bcbd22",  # yellow
        "SSOLVED": "#bcbd22",  # yellow (typo variant)
    }
    return mapping.get(status, "#888888")


# ===========================================================================
# VISUALIZATION: UPSET PLOT
# ===========================================================================


def create_upsetplot(
        models_data: List[Tuple[str, List[dict]]],
        data_selector_func,
        plot_title: Optional[str] = "",
):
    """
    Create an UpSet plot to visualize set intersections.

    Args:
        models_data: List of tuples (model_label, results)
        data_selector_func: Function to filter data (_ok_task_ids or _failed_task_ids)
        plot_title: Optional title for the plot

    Returns:
        Matplotlib Figure object or None if no data is available
    """
    data_sets = {label: data_selector_func(res) for label, res in models_data}
    all_ids = set().union(*data_sets.values())

    memberships = [tuple(lbl for lbl, s in data_sets.items() if tid in s) for tid in all_ids]

    if not memberships:
        return None

    data = from_memberships(memberships)
    labels = list(data_sets.keys())
    unique_combos = list({tuple(sorted(mem)) for mem in memberships if len(mem) > 1})

    fig = PLT.figure()
    u = UpSet(data, subset_size="count", show_counts=True)

    model_colors = _build_specific_model_colors(labels)

    for lbl in labels:
        color = model_colors.get(lbl, "#7f7f7f")
        u.style_categories([lbl], bar_facecolor=color, bar_edgecolor=color)

    for lbl in labels:
        color = model_colors.get(lbl, "#808080")
        u.style_subsets(present=lbl, min_degree=1, max_degree=1, facecolor=color)

    for combo in unique_combos:
        color = _intersection_color_for(combo, labels)
        u.style_subsets(
            present=list(combo),
            min_degree=len(combo),
            max_degree=len(combo),
            facecolor=color,
        )

    u.plot(fig=fig)

    if plot_title:
        fig.suptitle(plot_title, fontsize=12)

    return fig


# ===========================================================================
# VISUALIZATION: SANKEY DIAGRAM
# ===========================================================================


def create_sankey_status_flow_interactive(
        new_results: List[Dict[str, Any]],
        competitor_results: List[Dict[str, Any]],
        new_label: str,
        competitor_label: str,
):
    """Create interactive Sankey diagram showing status flow per turn."""

    def build_flow(turn_idx: int) -> Dict[Tuple[str, str], int]:
        """Build flow counts for a specific turn."""
        comp_map = {}
        for item in competitor_results:
            tid = _extract_task_id(item)
            if tid is not None:
                res = item.get("test_results", [])
                status = res[turn_idx].get("status") if 0 <= turn_idx < len(res) else "SOLVED"
                comp_map[tid] = status

        new_map = {}
        for item in new_results:
            tid = _extract_task_id(item)
            if tid is not None:
                res = item.get("test_results", [])
                status = res[turn_idx].get("status") if 0 <= turn_idx < len(res) else "SOLVED"
                new_map[tid] = status

        common = set(comp_map) & set(new_map)
        flows: Dict[Tuple[str, str], int] = {}

        for tid in common:
            left_status = comp_map.get(tid, "SOLVED")
            right_status = new_map.get(tid, "SOLVED")
            if left_status == "SOLVED" and right_status == "SOLVED":
                continue
            pair = (left_status, right_status)
            flows[pair] = flows.get(pair, 0) + 1
        return flows

    new_turns = {t for item in new_results for t in range(len(item.get("test_results", [])))}
    comp_turns = {
        t for item in competitor_results for t in range(len(item.get("test_results", [])))
    }
    all_turns = sorted(new_turns | comp_turns)
    if not all_turns:
        return None

    flows_by_turn = {t: build_flow(t) for t in all_turns}
    agg_flows: Dict[Tuple[str, str], int] = {}
    for f in flows_by_turn.values():
        for k, v in f.items():
            agg_flows[k] = agg_flows.get(k, 0) + v

    first_flows = agg_flows if agg_flows else flows_by_turn.get(all_turns[0], {})
    left = sorted({a for a, _ in first_flows})
    right = sorted({b for _, b in first_flows})

    labels = [f"{s} ({competitor_label})" for s in left] + [f"{s} ({new_label})" for s in right]
    left_idx = {s: i for i, s in enumerate(left)}
    right_idx = {s: i + len(left) for i, s in enumerate(right)}

    status_labels = sorted(set(left) | set(right))
    status_color_map = {s: _status_to_color(s) for s in status_labels}

    def make_trace(flows: Dict[Tuple[str, str], int]):
        """Create trace data for Sankey diagram."""
        sources, targets, values, colors = [], [], [], []
        for (a, b), v in flows.items():
            sources.append(left_idx.get(a, 0))
            targets.append(right_idx.get(b, 0))
            values.append(v)
            colors.append(status_color_map.get(b, "#888888"))
        return dict(source=sources, target=targets, value=values, color=colors)

    node_colors = [status_color_map[s] for s in left] + [status_color_map[s] for s in right]
    fig = GO.Figure(
        GO.Sankey(
            node=dict(label=labels, pad=20, thickness=20, color=node_colors),
            link=make_trace(first_flows),
        )
    )

    buttons = []

    buttons.append(
        dict(
            method="update",
            label="All Turns",
            args=[
                {"link": [make_trace(agg_flows)]},
                {
                    "title": {
                        "text": f"Status Flow from {competitor_label} to {new_label} • All Turns"
                    }
                },
            ],
        )
    )

    for t in all_turns:
        flows = build_flow(t)
        buttons.append(
            dict(
                method="update",
                label=f"Turn {t + 1}",
                args=[
                    {"link": [make_trace(flows)]},
                    {
                        "title": {
                            "text": f"Status Flow from {competitor_label} to {new_label} • Turn {t + 1}"
                        }
                    },
                ],
            )
        )

    fig.update_layout(
        title=f"Status Flow from ({competitor_label} to {new_label}) • All Turns",
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="right",
                x=0.1,
                xanchor="left",
                y=0.1,
                yanchor="bottom",
            )
        ],
        height=550,
        hoverlabel=dict(bgcolor="white"),
        margin=dict(b=80),
    )

    return fig


# ===========================================================================
# VISUALIZATION: MULTI-TURN SUCCESS BAR CHART
# ===========================================================================


def create_multiturn_success_bar_multi(models_data: List[Tuple[str, List[Dict[str, Any]]]]):
    """Create stacked percentage bar chart showing successful tests per turn per model."""
    ok_counts = {}
    model_labels = []
    total_counts = {}
    sum_labels = []

    for label, res in models_data:
        per_turn = {}
        total = len(res)
        for item in res:
            if item.get("test_status") == "OK":
                turn = int(item.get("turn", 0))
                per_turn[turn] = per_turn.get(turn, 0) + 1
        ok_counts[label] = per_turn
        total_counts[label] = total
        model_labels.append(label)

    all_turns = {t for counts in ok_counts.values() for t in counts}
    turns = sorted(all_turns)

    fig = GO.Figure()
    for t in turns:
        y_vals = []
        for m in model_labels:
            count = ok_counts.get(m, {}).get(t, 0)
            total = total_counts.get(m, 0)
            percentage = (count / total) if total > 0 else 0
            y_vals.append(percentage)

        fig.add_trace(GO.Bar(name=f"Turn {t}", x=model_labels, y=y_vals, hovertemplate="%{y:.1%}"))

    # Calculate the sum of stacked bar labels for all models.
    for m in model_labels:
        total_sum = 0
        for t in turns:
            total_sum += ok_counts.get(m, {}).get(t, 0) / total_counts.get(m, 1)
        sum_labels.append(f"{total_sum * 100:.1f}%")

    # Update plot layout.
    fig.update_layout(
        barmode="stack",
        height=500,
        showlegend=True,
        legend_title="Turn",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2, yanchor="top"),
        margin=dict(b=100),
        xaxis=dict(title="Checkpoint"),
        yaxis=dict(
            title="Success Rate (%)",
            tickmode="linear",
            dtick=0.2,
            range=[0, 1],
            tickformat=".0%",
            ticksuffix="",
        ),
    )

    for i, label in enumerate(sum_labels):
        fig.add_annotation(
            x=model_labels[i],
            y=-0.05,
            text=label,
            showarrow=False,
            font=dict(size=12),
            align="center",
        )

    return fig


# ===========================================================================
# VISUALIZATION: ERROR PIE CHART
# ===========================================================================


def create_error_pie_interactive(res: List[Dict[str, Any]]):
    """Create interactive pie chart of error types by turn."""
    error_data = {}

    for item in res:
        turn_results = item.get("test_results", [])
        for turn_idx, res in enumerate(turn_results):
            err_type = _extract_error_type(item, True, turn_idx)
            turn = turn_idx + 1
            if turn not in error_data:
                error_data[turn] = []
            error_data[turn].append(err_type)

    unique_turns = sorted(error_data.keys())

    def _top7_with_other(error_types: List[str]) -> Tuple[List[str], List[int]]:
        """Get top 7 error types and group rest as 'Others'."""

        counts = {}
        for err_type in error_types:
            counts[err_type] = counts.get(err_type, 0) + 1

        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        # Take top 7
        top_items = sorted_items[:7]
        other_items = sorted_items[7:]

        labels = [item[0] for item in top_items]
        values = [item[1] for item in top_items]

        other_sum = sum(item[1] for item in other_items)
        if other_sum > 0:
            labels.append("Others")
            values.append(other_sum)

        return labels, values

    all_error_types = []
    for turn_errors in error_data.values():
        all_error_types.extend(turn_errors)

    labels_all, values_all = _top7_with_other(all_error_types)
    fig = GO.Figure(
        GO.Pie(
            labels=labels_all, values=values_all, sort=False, textinfo="label+percent", hole=0.35
        )
    )

    buttons = []
    buttons.append(
        dict(
            method="update",
            label="All Turns",
            args=[
                {"labels": [labels_all], "values": [values_all]},
                {"title": {"text": "Error types (all turns)"}},
            ],
        )
    )

    for t in unique_turns:
        turn_error_types = error_data[t]
        labels, values = _top7_with_other(turn_error_types)
        buttons.append(
            dict(
                method="update",
                label=f"Turn {t}",
                args=[
                    {"labels": [labels], "values": [values]},
                    {"title": {"text": f"Error types (turn {t})"}},
                ],
            )
        )

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=1.15, y=1)],
        height=500,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2, yanchor="top"),
        margin=dict(b=100),
    )

    return fig


# ===========================================================================
# VISUALIZATION: LEVENSHTEIN DISTANCE BOXPLOT
# ===========================================================================


def create_levenshtein_boxplot_interactive(
        models_data: List[Tuple[str, Dict[Union[int, str], Dict[str, List[int]]]]],
):
    """
    Create interactive boxplot of Levenshtein distances across models with turn/status selectors.

    Args:
        models_data: List of (model_label, {turn -> {status -> [distances]}}) tuples

    Returns:
        Plotly Figure object or None if no data is available
    """

    def _normalize_turn_status_mapping(
            raw: Dict[Union[int, str], Union[Dict[str, List[int]], List[int]]],
    ) -> Dict[Union[int, str], Dict[str, List[int]]]:
        """Normalize mapping to ensure consistent structure."""
        # Case 1: {turn(int) -> {status -> list}}
        has_int_turns = any(isinstance(k, int) for k in raw.keys())
        values_are_dicts = all(isinstance(v, dict) for v in raw.values())

        if has_int_turns and values_are_dicts:
            out: Dict[Union[int, str], Dict[str, List[int]]] = {}
            agg: Dict[str, List[int]] = {}

            for k, v in raw.items():
                if isinstance(k, int) and isinstance(v, dict):
                    out[k] = {str(st): list(vals) for st, vals in v.items()}
                    for st, vals in v.items():
                        agg.setdefault(str(st), []).extend(list(vals))

            out["ALL"] = agg
            return out

        # Case 2: {status -> list} aggregate
        if all(not isinstance(k, int) for k in raw.keys()):
            agg = {str(st): list(vals) for st, vals in raw.items() if isinstance(vals, list)}
            return {"ALL": agg}

        return {"ALL": {}}

    normalized: List[Tuple[str, Dict[Union[int, str], Dict[str, List[int]]]]] = [
        (label, _normalize_turn_status_mapping(mapping)) for label, mapping in models_data
    ]

    all_turns: Set[int] = set()
    all_statuses: Set[str] = set()

    for _, turn_to_status in normalized:
        for tk, sm in turn_to_status.items():
            if isinstance(tk, int):
                all_turns.add(tk)
            for st in sm.keys():
                all_statuses.add(str(st))

    turns = sorted(all_turns)

    def _has_any_values_for_status(status_key: str) -> bool:
        for _, turn_to_status in normalized:
            if "ALL" in turn_to_status and list(turn_to_status["ALL"].get(status_key, [])):
                return True
            for tk, sm in turn_to_status.items():
                if isinstance(tk, int) and list(sm.get(status_key, [])):
                    return True
        return False

    statuses = sorted([s for s in all_statuses if _has_any_values_for_status(s)])

    if not statuses:
        return None

    def build_model_traces(
            turn_option: Optional[int], include_overall: bool = True
    ) -> List[Tuple[List[str], List[int]]]:
        """Build traces for each model (models in legend, status on x-axis)."""
        traces = []
        status_categories = statuses + (["Overall"] if include_overall else [])

        for label, turn_to_status in normalized:
            xs: List[str] = []
            ys: List[int] = []

            for s in status_categories:
                vals: List[int] = []

                if s == "Overall":
                    # For "Overall" status, aggregate all statuses for this model
                    if turn_option is None:
                        aggregated_any = False
                        for tk, sm in turn_to_status.items():
                            if isinstance(tk, int):
                                for status_vals in sm.values():
                                    vals.extend(list(status_vals))
                                aggregated_any = True

                        if not aggregated_any and "ALL" in turn_to_status:
                            for status_vals in turn_to_status.get("ALL", {}).values():
                                vals.extend(list(status_vals))
                    else:
                        sm = turn_to_status.get(turn_option, {})
                        for status_vals in sm.values():
                            vals.extend(list(status_vals))
                else:
                    # For specific status
                    if turn_option is None:
                        aggregated_any = False
                        for tk, sm in turn_to_status.items():
                            if isinstance(tk, int):
                                vals.extend(list(sm.get(s, [])))
                                aggregated_any = True

                        if not aggregated_any and "ALL" in turn_to_status:
                            vals = list(turn_to_status.get("ALL", {}).get(s, []))
                    else:
                        vals = list(turn_to_status.get(turn_option, {}).get(s, []))

                xs.extend([s] * len(vals))
                ys.extend(vals)

            traces.append((xs, ys))

        return traces

    model_labels = [label for label, _ in normalized]

    fig = GO.Figure()
    base_traces = build_model_traces(turn_option=None, include_overall=True)

    for model_label, (xs, ys) in zip(model_labels, base_traces):
        fig.add_trace(GO.Box(x=xs, y=ys, name=str(model_label), boxmean=True))

    buttons = []

    if turns:
        all_x = [t[0] for t in base_traces]
        all_y = [t[1] for t in base_traces]
        buttons.append(
            dict(
                method="update",
                label="All Turns",
                args=[
                    {"x": all_x, "y": all_y},
                    {"title": {"text": "Levenshtein Distance by Status (All Turns)"}},
                ],
            )
        )

        for t in turns:
            traces_t = build_model_traces(turn_option=t, include_overall=True)
            btn_x = [tup[0] for tup in traces_t]
            btn_y = [tup[1] for tup in traces_t]
            buttons.append(
                dict(
                    method="update",
                    label=f"Turn {t}",
                    args=[
                        {"x": btn_x, "y": btn_y},
                        {"title": {"text": f"Levenshtein Distance by Status (Turn {t})"}},
                    ],
                )
            )

    layout_config = {
        "title": "Levenshtein Distance by Status (All Turns)",
        "yaxis_title": "Distance",
        "boxmode": "group",
        "height": 550,
        "legend": dict(orientation="h", x=0.5, xanchor="center", y=-0.15, yanchor="top"),
        "margin": dict(b=120),
    }

    if buttons:
        layout_config["updatemenus"] = [
            dict(
                type="buttons",
                direction="up",
                active=0,
                buttons=buttons,
                x=-0.05,
                xanchor="left",
                y=0.0,
                yanchor="bottom",
                showactive=True,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
            )
        ]
        layout_config["margin"] = dict(l=150, b=120)

    fig.update_layout(**layout_config)

    fig.update_layout(
        updatemenus=(
                layout_config.get("updatemenus", [])
                + [
                    dict(
                        type="buttons",
                        direction="up",
                        buttons=[
                            dict(label="Linear", method="relayout", args=[{"yaxis.type": "linear"}]),
                            dict(label="Log", method="relayout", args=[{"yaxis.type": "log"}]),
                        ],
                        x=1.05,
                        xanchor="right",
                        y=0.0,
                        yanchor="bottom",
                    )
                ]
        )
    )

    return fig


# ===========================================================================
# WANDB INTEGRATION
# ===========================================================================


def build_wandb_table_for_results(res: List[Dict[str, Any]]) -> wandb.Table:
    """Build a W&B table with all turns and codes."""
    logger.debug(f"Building W&B table for {len(res)} results")
    columns = [
        "task_id",
        "turn",
        "test_status",
        "error_type",
        "prompt",
        "ground_truth",
        "generated_code",
        "raw_response",
    ]
    table = wandb.Table(columns=columns)

    total_rows = 0
    for item in res:
        test_results = item.get("test_results", [])
        codes = item.get("generate_codes", [])
        num_turns = max(len(test_results), len(codes))

        for turn_idx in range(num_turns):
            status = test_results[turn_idx].get("status") if turn_idx < len(test_results) else None
            table.add_data(
                _extract_task_id(item),
                turn_idx + 1,
                status,
                _extract_error_type(item, False, turn_idx),
                _extract_prompt(item),
                _extract_ground_truth_code(item),
                _extract_generated_code(item, turn_idx),
                _extract_raw_response(item, turn_idx),
            )
            total_rows += 1

    logger.info(f"Built table with {total_rows} rows from {len(res)} results")
    return table


def log_results_and_graphs(
        wandb_config: Dict[str, Any],
        results_by_name: Dict[str, List[Dict[str, Any]]],
        run_label: str,
        analysis_cfg: Optional[Dict[str, Any]],
) -> None:
    """
    Log W&B tables, charts, and comparisons.

    Args:
        wandb_config: W&B configuration dictionary
        results_by_name: Dictionary mapping result names to result lists
        run_label: Label for the current run
        analysis_cfg: Optional analysis configuration with competitors info
    """
    for name, res in results_by_name.items():
        logger.info(f"  - {name}: {len(res)} results")

    log_payload: Dict[str, Any] = {}

    new_label = (analysis_cfg or {}).get("new_model_label", run_label)
    new_results = results_by_name.get(run_label) or next(iter(results_by_name.values()))

    log_payload[f"Table_of_results_for_{run_label}"] = build_wandb_table_for_results(new_results)
    logger.info("Prepared table for current checkpoint results")

    # 2. Error pie chart
    if pie := create_error_pie_interactive(new_results):
        log_payload["Error types"] = wandb.Plotly(pie)
        logger.info("Prepared error pie chart for current checkpoint")

    # Load competitors or use neighboring checkpoints for comparison
    competitors_cfg = (analysis_cfg or {}).get("competitors", [])
    competitors = _load_competitors_once(competitors_cfg)

    def extract_checkpoint_num(name: str) -> int:
        """Extract checkpoint number from name like 'checkpoint-478' or just '478'"""
        import re
        match = re.search(r'(\d+)$', name)
        return int(match.group(1)) if match else 0

    # If no competitors but multiple checkpoints, use neighboring checkpoints for comparison
    if not competitors and len(results_by_name) > 1:
        sorted_checkpoints = sorted(results_by_name.items(), key=lambda x: extract_checkpoint_num(x[0]))
        current_idx = next((i for i, (lbl, _) in enumerate(sorted_checkpoints) if lbl == run_label), 0)

        comparison_data = []
        if current_idx > 0:
            comparison_data.append(sorted_checkpoints[current_idx - 1])
        elif current_idx == 0 and len(sorted_checkpoints) > 2:
            comparison_data.append(sorted_checkpoints[1])
        comparison_data.append((run_label, new_results))
        if current_idx < len(sorted_checkpoints) - 1:
            comparison_data.append(sorted_checkpoints[current_idx + 1])
        elif current_idx == len(sorted_checkpoints) - 1 and len(sorted_checkpoints) > 2:
            comparison_data.append(sorted_checkpoints[0])
        logger.info(f"Comparing {len(comparison_data)} checkpoints: {[lbl for lbl, _ in comparison_data]}")
    else:
        comparison_data = [(new_label, new_results)] + [(lbl, res) for lbl, res in competitors]
        logger.info(f"Loaded {len(competitors)} competitors")

    # 3. UpSet diagrams (success and failed)
    if len(comparison_data) > 1:
        fig = create_upsetplot(
            models_data=comparison_data,
            data_selector_func=lambda res: _filter_task_ids_by_status(res, "OK"),
            plot_title="Success intersections (UpSet)",
        )
        if fig is not None:
            log_payload["Intersections of successful tests"] = wandb.Image(fig)
            logger.info("Prepared Upset plot for successful tests")

        fig = create_upsetplot(
            models_data=comparison_data,
            data_selector_func=lambda res: _filter_task_ids_by_status(res, "FAILED"),
            plot_title="Failure intersections (UpSet)",
        )
        if fig is not None:
            log_payload["Intersections of failed tests"] = wandb.Image(fig)
            logger.info("Prepared Upset plot for failed tests")

    # Sankey flows (for neighbors/competitors) - exclude current checkpoint
    neighbors_for_sankey = [(lbl, res) for lbl, res in comparison_data if lbl != run_label]
    for idx, (comp_label, comp_res) in enumerate(neighbors_for_sankey):
        sankey = create_sankey_status_flow_interactive(
            new_results, comp_res, new_label, comp_label
        )
        if idx == 0:
            log_payload["Test_status_flow_for_first_neighbor"] = wandb.Plotly(sankey)
            logger.info(f"Prepared Sankey flow for {comp_label}")
        elif idx == 1:
            log_payload["Test_status_flow_for_second_neighbor"] = wandb.Plotly(sankey)
            logger.info(f"Prepared Sankey flow for {comp_label}")

    # 4. Multi-turn bar chart
    multi = create_multiturn_success_bar_multi(comparison_data)
    log_payload["Multiturn success"] = wandb.Plotly(multi)
    logger.info("Prepared multi-turn success bar chart")

    # 6. Levenshtein plots
    models_for_lev_by_turn = [(lbl, _collect_lev_rows_by_turn(res)) for lbl, res in comparison_data]

    lev_fig = create_levenshtein_boxplot_interactive(models_for_lev_by_turn)
    if lev_fig is not None:
        log_payload["Levenshtein distance by status"] = wandb.Plotly(lev_fig)
        logger.info("Prepared Levenshtein boxplot")

    if log_payload:
        wandb.log(log_payload)
        logger.info(f"Logged {len(log_payload)} media items in a single call")

    logger.info("Finished W&B plotting and logging")
