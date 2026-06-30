import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { GraphExplorer } from "./GraphExplorer";

describe("GraphExplorer", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("renderiza os controles Forward/Backward e os sliders", () => {
    render(<GraphExplorer />);
    expect(screen.getByRole("button", { name: /forward/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /backward/i })).toBeInTheDocument();
    expect(screen.getAllByRole("slider").length).toBeGreaterThanOrEqual(3);
  });

  it("clicar em Backward não quebra a renderização", () => {
    render(<GraphExplorer />);
    fireEvent.click(screen.getByRole("button", { name: /backward/i }));
    expect(screen.getByRole("button", { name: /backward/i })).toBeInTheDocument();
  });

  it("após Backward e animação completa, exibe gradientes não-zero", async () => {
    vi.useFakeTimers();
    render(<GraphExplorer />);

    // Before backward: all grads should be 0
    expect(
      screen.getByTestId("has-nonzero-grads").getAttribute("data-value"),
    ).toBe("false");

    fireEvent.click(screen.getByRole("button", { name: /backward/i }));

    // Advance timers well past the full animation (7 frames × 300 ms = 2100 ms)
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });

    // After animation completes, at least the root node has grad = 1
    expect(
      screen.getByTestId("has-nonzero-grads").getAttribute("data-value"),
    ).toBe("true");
  });

  it("Forward após Backward reseta para grads zero", async () => {
    vi.useFakeTimers();
    render(<GraphExplorer />);

    fireEvent.click(screen.getByRole("button", { name: /backward/i }));
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });

    // Should now show non-zero grads
    expect(
      screen.getByTestId("has-nonzero-grads").getAttribute("data-value"),
    ).toBe("true");

    // Click Forward — should reset to all-zero grads
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /forward/i }));
    });

    expect(
      screen.getByTestId("has-nonzero-grads").getAttribute("data-value"),
    ).toBe("false");
  });
});
