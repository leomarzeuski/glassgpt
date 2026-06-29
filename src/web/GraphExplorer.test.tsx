import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { GraphExplorer } from "./GraphExplorer";

describe("GraphExplorer", () => {
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
});
