import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TrainingDemo } from "./TrainingDemo";

describe("TrainingDemo", () => {
  it("mostra a loss após um passo", () => {
    render(<TrainingDemo />);
    expect(screen.getByTestId("loss")).toHaveTextContent("—");
    fireEvent.click(screen.getByRole("button", { name: /1 passo/i }));
    expect(screen.getByTestId("loss")).not.toHaveTextContent("—");
  });
});
