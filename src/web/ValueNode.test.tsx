import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { ValueNode, type ValueNodeData } from "./ValueNode";

const renderNode = (data: ValueNodeData) =>
  render(
    <ReactFlowProvider>
      <ValueNode data={data} />
    </ReactFlowProvider>,
  );

describe("ValueNode", () => {
  it("mostra valor e gradiente", () => {
    renderNode({ label: "x", value: 1.2345, grad: 0.5, op: "" });
    expect(screen.getByText(/1\.2345/)).toBeInTheDocument();
    expect(screen.getByText(/0\.5000/)).toBeInTheDocument();
  });

  it("marca nós com valor não-finito", () => {
    renderNode({ label: "", value: Infinity, grad: 0, op: "/" });
    expect(screen.getByTestId("value-node")).toHaveAttribute("data-nonfinite", "true");
  });
});
