import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { GraphExplorer } from "./GraphExplorer";

// Reads "gradientes calculados: X/Y" -> [revealed, total]
function gradCount(): [number, number] {
  const text = screen.getByTestId("grad-status").textContent ?? "";
  const m = text.match(/(\d+)\/(\d+)/);
  if (!m) throw new Error(`status text sem contador: "${text}"`);
  return [Number(m[1]), Number(m[2])];
}

// The animation reschedules a timer inside an effect after each frame, so we
// advance one interval at a time, flushing React between ticks.
async function playAnimation(ticks = 12) {
  for (let i = 0; i < ticks; i++) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(400);
    });
  }
}

describe("GraphExplorer", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("renderiza os controles e os sliders", () => {
    render(<GraphExplorer />);
    expect(screen.getByRole("button", { name: /animar backward/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /resetar/i })).toBeInTheDocument();
    expect(screen.getAllByRole("slider").length).toBeGreaterThanOrEqual(3);
  });

  it("começa com 0 gradientes calculados (estado forward)", () => {
    render(<GraphExplorer />);
    const [revealed, total] = gradCount();
    expect(revealed).toBe(0);
    expect(total).toBeGreaterThan(0);
  });

  it("após Animar backward e a animação completar, todos os gradientes são calculados", async () => {
    vi.useFakeTimers();
    render(<GraphExplorer />);

    fireEvent.click(screen.getByRole("button", { name: /animar backward/i }));
    await playAnimation();

    const [revealed, total] = gradCount();
    expect(revealed).toBe(total);
    expect(revealed).toBeGreaterThan(0);
  });

  it("Resetar após o backward volta para 0 gradientes", async () => {
    vi.useFakeTimers();
    render(<GraphExplorer />);

    fireEvent.click(screen.getByRole("button", { name: /animar backward/i }));
    await playAnimation();
    expect(gradCount()[0]).toBeGreaterThan(0);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /resetar/i }));
    });

    expect(gradCount()[0]).toBe(0);
  });
});
