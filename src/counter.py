"""
Bilateral vehicle counter module.
Accurately records vehicle traffic volume across directional gates and classifies traffic composition.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Set

from src.schemas import CountingGate, TrackedVehicle

logger = logging.getLogger(__name__)


class VehicleCounter:
    """Manages multi-gate directional traffic counting and class aggregation."""

    def __init__(self, gates: Optional[Dict[str, CountingGate]] = None) -> None:
        """
        Initialize the vehicle counter.

        Args:
            gates: Dictionary mapping gate keys (e.g. 'left_gate', 'right_gate') to CountingGate objects.
        """
        self.gates: Dict[str, CountingGate] = gates or {}
        # Breakdown by vehicle class: {"car": 12, "truck": 3, ...}
        self.class_breakdown: Dict[str, int] = {}
        # Gate triggers for visual highlighting in the current frame
        self.active_gate_triggers: Set[str] = set()

    def add_gate(self, key: str, gate: CountingGate) -> None:
        """Register a new counting gate."""
        self.gates[key] = gate

    def update(self, vehicles: Sequence[TrackedVehicle]) -> Dict[str, int]:
        """
        Evaluate positions of tracked vehicles against counting gates.

        Args:
            vehicles: Current frame active vehicles.

        Returns:
            Dictionary mapping gate keys to their total count so far.
        """
        self.active_gate_triggers.clear()

        for vehicle in vehicles:
            cx, cy = vehicle.centroid

            for gate_key, gate in self.gates.items():
                if vehicle.track_id in gate.counted_ids:
                    continue

                x_min = min(gate.line_up[0], gate.line_down[0], gate.line_up[2], gate.line_down[2])
                x_max = max(gate.line_up[0], gate.line_down[0], gate.line_up[2], gate.line_down[2])
                y_min = min(gate.line_up[1], gate.line_up[3])
                y_max = max(gate.line_down[1], gate.line_down[3])

                # Check if centroid falls within the counting zone between boundary lines
                if x_min <= cx <= x_max and y_min <= cy <= y_max:
                    gate.counted_ids.add(vehicle.track_id)
                    gate.count += 1
                    vehicle.is_counted = True
                    self.active_gate_triggers.add(gate_key)

                    # Update classification metrics
                    c_name = vehicle.class_name
                    self.class_breakdown[c_name] = self.class_breakdown.get(c_name, 0) + 1

        return {k: g.count for k, g in self.gates.items()}

    def get_counts(self) -> Dict[str, int]:
        """Return the current counts for all registered gates."""
        return {k: g.count for k, g in self.gates.items()}

    def get_total_count(self) -> int:
        """Return aggregate vehicle volume across all gates."""
        return sum(g.count for g in self.gates.values())

    def is_gate_active(self, gate_key: str) -> bool:
        """Check if a gate was triggered during the most recent frame."""
        return gate_key in self.active_gate_triggers
