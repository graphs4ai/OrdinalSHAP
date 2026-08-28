import matplotlib.pyplot as plt
import numpy as np


class ShapleyVisualization:

  POSITIVE_COLOR = "#2b6cb0"
  NEGATIVE_COLOR = "#c53030"

  def __init__(self, tokens, values, vmax=None, label=None):
    self.tokens = list(tokens)
    self.values = np.asarray(values, dtype=float)
    self.label = label
    self.vmax = vmax if vmax is not None else (
      float(np.max(np.abs(self.values))) if len(self.values) else 0.0
    )

  def _alpha_for(self, value):
    if self.vmax <= 0:
      return 0.0
    return min(abs(value) / self.vmax, 1.0)

  def plot_bar_chart(self, output_path, top_k=10, figsize=None, dpi=200):
    order = np.argsort(-np.abs(self.values))

    if top_k is not None and len(order) > top_k:
      top_idx, rest_idx = order[:top_k], order[top_k:]
    else:
      top_idx, rest_idx = order, np.array([], dtype=int)

    labels = [self.tokens[i] for i in top_idx]
    values = [self.values[i] for i in top_idx]

    if len(rest_idx) > 0:
      labels.append(f"Sum of {len(rest_idx)} other features")
      values.append(float(np.sum(self.values[rest_idx])))

    labels, values = labels[::-1], values[::-1]
    colors = [self.POSITIVE_COLOR if v >= 0 else self.NEGATIVE_COLOR for v in values]

    if figsize is None:
      figsize = (7, 0.45 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, height=0.65, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="#333333", linewidth=0.8, zorder=2)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)

    for spine in ("top", "right", "left"):
      ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#888888")

    xmax = max((abs(v) for v in values), default=1.0) * 1.3
    ax.set_xlim(-xmax, xmax)
    offset = xmax * 0.02
    for y, v in zip(y_pos, values):
      ha = "left" if v >= 0 else "right"
      x_text = v + offset if v >= 0 else v - offset
      color = self.POSITIVE_COLOR if v >= 0 else self.NEGATIVE_COLOR
      ax.text(x_text, y, f"{v:+.3f}", va="center", ha=ha, fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel("Shapley value", fontsize=10)
    if self.label:
      ax.set_title(self.label, fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

  def plot_text(self, output_path, width_in=10, dpi=200, fontsize=13, line_width_frac=0.94, show_values=True):
    probe_fig = plt.figure(figsize=(width_in, 1), dpi=dpi)
    probe_ax = probe_fig.add_axes([0, 0, 1, 1])
    probe_ax.axis("off")
    probe_fig.canvas.draw()
    renderer = probe_fig.canvas.get_renderer()

    fig_width_px = width_in * dpi
    max_x_px = fig_width_px * line_width_frac

    def rendered_width(s, size=fontsize):
      t = probe_ax.text(0, 0, s, fontsize=size, family="sans-serif")
      width = t.get_window_extent(renderer=renderer).width
      t.remove()
      return width

    space_width = rendered_width(" ")

    box_pad_px = fontsize * (dpi / 72.0) * 0.35

    positions = []
    x_px, line = 0.0, 0
    for token, value in zip(self.tokens, self.values):
      token_width = rendered_width(token)
      full_width = token_width + 2 * box_pad_px
      if x_px > 0 and x_px + full_width > max_x_px:
        line += 1
        x_px = 0.0
      positions.append((token, value, x_px, line, token_width))
      x_px += full_width + space_width

    plt.close(probe_fig)
    n_lines = line + 1

    line_units = 1.7 if show_values else 1.0
    title_rows = 1.1 if self.label else 0.0
    fig_height_in = (n_lines * line_units + title_rows) * (fontsize / 72.0) * 2.0

    fig = plt.figure(figsize=(width_in, fig_height_in), dpi=dpi)
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.axis("off")
    ax.set_xlim(0, fig_width_px)
    ax.set_ylim(0, n_lines * line_units)
    ax.invert_yaxis()

    for token, value, x_px, line, token_width in positions:
      alpha = self._alpha_for(value)
      color = self.POSITIVE_COLOR if value >= 0 else self.NEGATIVE_COLOR
      row_y = line * line_units
      ax.text(
        x_px + box_pad_px, row_y + line_units * 0.32, token,
        fontsize=fontsize, va="center", ha="left", family="sans-serif",
        bbox=dict(facecolor=color, alpha=0.15 + 0.85 * alpha, edgecolor="none", boxstyle="round,pad=0.2"),
      )
      if show_values:
        word_center_x = x_px + box_pad_px + token_width / 2
        ax.text(
          word_center_x, row_y + line_units * 0.75, f"{value:+.3f}",
          fontsize=fontsize * 0.62, va="center", ha="center", family="sans-serif",
          color=color, fontweight="bold",
        )

    if self.label:
      ax.set_title(self.label, fontsize=13, loc="left", fontweight="bold")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path