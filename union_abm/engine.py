"""
Union ABM Core Simulation Engine

This module implements the core simulation logic for the union organizing model,
including the Stochastic Block Model network, Agent dynamics, and strategy logic.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


class AgentState(Enum):
    """Agent state enumeration following union organizing framework."""

    PASSIVE = 0  # Not engaged, susceptible to mobilization
    MOBILIZED = 1  # Engaged but not yet committed organizer
    ORGANIZER = 2  # Active organizer recruiting others


@dataclass
class Agent:
    """
    Agent class representing a worker in the organizing network.

    Attributes:
        agent_id: Unique identifier for the agent
        energy: Float in [-0.5, 1.0] representing engagement level
        state: Current state (Passive, Mobilized, or Organizer)
        department_id: Department/block membership in the SBM
        persistence_counter: Ticks above threshold while being targeted
        demotion_counter: Ticks below omega threshold (for organizer demotion)
        is_targeted: Whether currently being targeted by an organizer
        targeting_organizer: ID of the organizer targeting this agent
        energy_min: Minimum possible energy (ideological floor)
        energy_max: Maximum possible energy (ideological ceiling)
    """

    agent_id: int
    energy: float
    state: AgentState = AgentState.PASSIVE
    department_id: int = 0
    persistence_counter: int = 0
    demotion_counter: int = 0  # Ticks below omega threshold (for organizer demotion)
    is_targeted: bool = False
    targeting_organizer: Optional[int] = None
    energy_min: float = -1.0  # Ideological range minimum
    energy_max: float = 1.0  # Ideological range maximum

    # Track history for analysis
    energy_history: List[float] = field(default_factory=list)
    state_history: List[AgentState] = field(default_factory=list)

    def __post_init__(self):
        """Initialize history with current values."""
        self.energy_history.append(self.energy)
        self.state_history.append(self.state)

    def clip_energy(self, max_threshold: Optional[float] = None) -> None:
        """
        Ensure energy stays within valid bounds based on ideological range.

        Args:
            max_threshold: Optional maximum energy threshold. If provided,
                          energy is clipped to min(energy_max, max_threshold)
        """
        max_energy = (
            self.energy_max
            if max_threshold is None
            else min(self.energy_max, max_threshold)
        )
        self.energy = np.clip(self.energy, self.energy_min, max_energy)

    def update_history(self) -> None:
        """Record current state to history."""
        self.energy_history.append(self.energy)
        self.state_history.append(self.state)


class UnionSim:
    """
    Union Organizing ABM Simulation.

    Implements a Stochastic Block Model network where agents can transition
    between Passive, Mobilized, and Organizer states based on energy dynamics
    and social influence.

    Parameters:
        n_departments: Number of departments (blocks in SBM)
        agents_per_dept: Number of agents per department
        p_in: Probability of intra-department connections
        p_out: Probability of inter-department connections
        seed_size: Number of initial organizers in the seed clique
        alpha: Strategy parameter [0,1] - probability of 1-to-1 vs broadcast
        beta: Complex contagion coefficient
        delta: Energy decay rate
        omega: Threshold for becoming an organizer
        mobilizing_threshold: Threshold for becoming mobilized (energy > threshold)
        persistence_threshold: Ticks above omega needed to become organizer
        outreach_energy: Energy boost from 1-to-1 organizing
        broadcast_energy: Energy boost from broadcast/mobilizing
        broadcast_reach_ratio: Fraction of agents reached by broadcast action
        contagion_power: Exponent for complex contagion term
        organizer_potential_ratio: Fraction of agents whose ideological range allows becoming organizer
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        n_departments: int = 5,
        agents_per_dept: int = 20,
        p_in: float = 0.4,
        p_out: float = 0.05,
        seed_size: int = 3,
        alpha: float = 0.8,
        beta: float = 0.3,
        delta: float = 0.01,
        omega: float = 0.8,
        mobilizing_threshold: float = 0.3,
        persistence_threshold: int = 3,
        outreach_energy: float = 0.5,
        broadcast_energy: float = 0.1,
        broadcast_reach_ratio: float = 0.1,
        contagion_power: float = 1.0,
        organizer_potential_ratio: float = 0.2,
        minimum_energy: float = -1.0,
        seed: Optional[int] = None,
    ):
        # Store parameters
        self.n_departments = n_departments
        self.agents_per_dept = agents_per_dept
        self.p_in = p_in
        self.p_out = p_out
        self.seed_size = seed_size
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.omega = omega
        self.mobilizing_threshold = mobilizing_threshold
        self.persistence_threshold = persistence_threshold
        self.outreach_energy = outreach_energy
        self.broadcast_energy = broadcast_energy
        self.broadcast_reach_ratio = broadcast_reach_ratio
        self.contagion_power = contagion_power
        self.organizer_potential_ratio = organizer_potential_ratio
        self.minimum_energy = minimum_energy
        self.seed = seed

        # Set random seeds
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Simulation state
        self.tick = 0
        self.agents: Dict[int, Agent] = {}
        self.graph: nx.Graph = None
        self.seed_clique: Set[int] = set()
        self.organizer_clique_edges: Set[Tuple[int, int]] = set()

        # Stop test state
        self.stop_test_active = False
        self.stop_test_tick: Optional[int] = None

        # Network snapshots for visualization
        self.snapshots: List[Dict] = []

        # Energy gradient tracking
        self.previous_total_energy: Optional[float] = None
        self.energy_components: Dict[str, float] = {
            "outreach": 0.0,
            "contagion": 0.0,
            "decay": 0.0,
        }

        # Initialize the simulation
        self._initialize_network()
        self._initialize_agents()
        self._initialize_seed_clique()

        # Store initial snapshot (tick 0)
        self._store_snapshot()

    def validate_parameter_balance(self) -> Dict[str, any]:
        """
        Validate parameter balance for fair strategy comparison.

        Checks:
        1. Total outreach energy is comparable between 1-to-1 and broadcast strategies
        2. Decay rate is in reasonable range compared to outreach energy

        Returns:
            Dictionary with 'balanced', 'warnings', and 'recommendations' keys
        """
        n_agents = len(self.agents)
        warnings = []
        recommendations = []

        # Calculate expected energy per tick at initialization
        # 1-to-1: each organizer gives outreach_energy to 1 agent
        total_1to1_energy = self.seed_size * self.outreach_energy

        # Broadcast: each organizer reaches broadcast_reach_ratio * n_agents
        # with broadcast_energy * n_organizers
        targets_per_organizer = int(n_agents * self.broadcast_reach_ratio)
        total_broadcast_energy = (
            self.seed_size * targets_per_organizer * self.broadcast_energy
        )

        # Check strategy balance (should be within factor of 3)
        ratio = (
            total_broadcast_energy / total_1to1_energy
            if total_1to1_energy > 0
            else float("inf")
        )

        if ratio > 3.0:
            warnings.append(
                f"⚠️  STRATEGY IMBALANCE: Broadcast provides {ratio:.1f}x more energy than 1-to-1"
            )
            # Suggest adjustment
            suggested_broadcast = total_1to1_energy / (
                self.seed_size * targets_per_organizer
            )
            recommendations.append(
                f"   Reduce broadcast_energy to ~{suggested_broadcast:.4f} for balance"
            )
        elif ratio < 0.33:
            warnings.append(
                f"⚠️  STRATEGY IMBALANCE: 1-to-1 provides {1/ratio:.1f}x more energy than broadcast"
            )
            suggested_broadcast = total_1to1_energy / (
                self.seed_size * targets_per_organizer
            )
            recommendations.append(
                f"   Increase broadcast_energy to ~{suggested_broadcast:.4f} for balance"
            )

        # Check decay vs outreach balance
        # Estimate initial decay: delta * n_agents * (1 - initial_engaged_fraction)
        # Assume initial engaged fraction is seed_size / n_agents
        initial_engaged_fraction = self.seed_size / n_agents
        estimated_total_decay = self.delta * n_agents * (1 - initial_engaged_fraction)

        # Compare decay to average outreach
        avg_outreach = (total_1to1_energy + total_broadcast_energy) / 2
        decay_ratio = (
            estimated_total_decay / avg_outreach if avg_outreach > 0 else float("inf")
        )

        if decay_ratio > 1.5:
            warnings.append(
                f"⚠️  DECAY OVERWHELMS OUTREACH: Decay is {decay_ratio:.1f}x larger than outreach energy"
            )
            suggested_delta = (
                avg_outreach * 0.8 / n_agents / (1 - initial_engaged_fraction)
            )
            recommendations.append(
                f"   Reduce delta to ~{suggested_delta:.4f} for balance"
            )
        elif decay_ratio < 0.1:
            warnings.append(
                f"⚠️  DECAY TOO WEAK: Decay is only {decay_ratio:.1%} of outreach energy"
            )
            suggested_delta = (
                avg_outreach * 0.5 / n_agents / (1 - initial_engaged_fraction)
            )
            recommendations.append(
                f"   Consider increasing delta to ~{suggested_delta:.4f} for meaningful decay"
            )

        return {
            "balanced": len(warnings) == 0,
            "warnings": warnings,
            "recommendations": recommendations,
            "metrics": {
                "total_1to1_energy": total_1to1_energy,
                "total_broadcast_energy": total_broadcast_energy,
                "strategy_ratio": ratio,
                "estimated_total_decay": estimated_total_decay,
                "decay_outreach_ratio": decay_ratio,
            },
        }

    def _initialize_network(self) -> None:
        """
        Create the Stochastic Block Model network.

        Uses NetworkX's stochastic_block_model to create a network with
        higher connectivity within departments than between them.
        """
        n_total = self.n_departments * self.agents_per_dept

        # Create block sizes (equal-sized departments)
        sizes = [self.agents_per_dept] * self.n_departments

        # Create probability matrix for SBM
        # p_in on diagonal, p_out off-diagonal
        probs = np.full((self.n_departments, self.n_departments), self.p_out)
        np.fill_diagonal(probs, self.p_in)

        # Generate SBM
        self.graph = nx.stochastic_block_model(sizes, probs.tolist(), seed=self.seed)

        # Store department assignments from the 'block' attribute
        self._department_map = {}
        for node in self.graph.nodes():
            self._department_map[node] = self.graph.nodes[node]["block"]

    def _initialize_agents(self) -> None:
        """
        Initialize agents with random energy values and ideological ranges.

        Each agent gets an ideological range (energy_min, energy_max) that defines
        their possible energy bounds. A configurable fraction (organizer_potential_ratio)
        of agents can potentially become organizers (energy_max >= omega).

        Initial energy is set randomly between -0.5 and mobilizing_threshold.
        """
        n_agents = len(self.graph.nodes())
        node_ids = list(self.graph.nodes())

        # Determine which agents can potentially become organizers
        n_potential_organizers = max(1, int(n_agents * self.organizer_potential_ratio))
        potential_organizer_ids = set(random.sample(node_ids, n_potential_organizers))

        for node_id in node_ids:
            # Assign ideological range based on potential organizer status
            if node_id in potential_organizer_ids:
                # Can become organizer: energy_max >= omega
                energy_min = np.random.normal(-0.3, 0.5)
                energy_max = 1.0
            else:
                # Cannot become organizer: energy_max < omega
                energy_min = np.random.normal(-0.3, 0.5)
                energy_max = np.random.uniform(
                    self.mobilizing_threshold, self.omega - 0.01
                )

            # Initial energy uniformly between -1 and mobilizing_threshold
            initial_energy = np.random.normal(energy_min + energy_max / 2, 0.3)
            initial_energy = np.clip(initial_energy, energy_min, energy_max)

            agent = Agent(
                agent_id=node_id,
                energy=initial_energy,
                state=AgentState.PASSIVE,
                department_id=self._department_map[node_id],
                energy_min=energy_min,
                energy_max=energy_max,
            )
            self.agents[node_id] = agent

    def _initialize_seed_clique(self) -> None:
        """
        Initialize the seed team clique of organizers.

        Selects seed_size agents from the first department to be the
        initial organizers, with high energy and full clique connectivity.
        """
        # Select agents from first department for seed clique
        dept_0_agents = [
            aid for aid, agent in self.agents.items() if agent.department_id == 0
        ]

        # Select seed_size agents (or all if fewer available)
        n_seeds = min(self.seed_size, len(dept_0_agents))
        seed_agents = random.sample(dept_0_agents, n_seeds)

        # Set them as organizers with high energy
        for aid in seed_agents:
            self.agents[aid].energy = 1.0
            self.agents[aid].state = AgentState.ORGANIZER
            self.seed_clique.add(aid)

        # Ensure full clique connectivity among seed organizers
        for i, aid1 in enumerate(seed_agents):
            for aid2 in seed_agents[i + 1 :]:
                if not self.graph.has_edge(aid1, aid2):
                    self.graph.add_edge(aid1, aid2)
                    self.organizer_clique_edges.add((min(aid1, aid2), max(aid1, aid2)))

    def get_neighbor_mean_energy(self, agent_id: int) -> float:
        """
        Calculate the mean energy of an agent's neighbors.

        Returns 0 if agent has no neighbors.
        """
        neighbors = list(self.graph.neighbors(agent_id))
        if not neighbors:
            return 0.0

        energies = [self.agents[n].energy for n in neighbors]
        return np.mean(energies)

    def get_engaged_neighbor_mean_energy(self, agent_id: int) -> float:
        """
        Calculate the mean energy of an agent's ENGAGED neighbors only.

        Only counts neighbors who are Mobilized or Organizer.
        Returns 0 if agent has no engaged neighbors (maximum decay effect).

        Sociological rationale: Only engaged peers provide social support
        that buffers against demobilization. Passive neighbors don't help, but count on social pressure.
        """
        neighbors = list(self.graph.neighbors(agent_id))
        if not neighbors:
            return 0.0

        # Filter to only mobilized and organizer neighbors
        engaged_neighbors = [
            n
            for n in neighbors
            if self.agents[n].state in (AgentState.MOBILIZED, AgentState.ORGANIZER)
        ]

        if not engaged_neighbors:
            return 0.0  # No engaged neighbors = maximum decay

        energies = [
            self.agents[n].energy for n in engaged_neighbors
        ]  # only engaged sum
        return np.sum(energies) / len(energies)

    def update_energy(
        self, agent_id: int, delta_outreach: float = 0.0
    ) -> Dict[str, float]:
        """
        Update agent energy following the union organizing dynamics.

        E_i(t+1) = clip(E_i(t) + ΔE_outreach + β(Ē_all)^p - δ(1 - Ē_engaged), -0.5, 1.0)

        Where:
        - ΔE_outreach: Energy from being targeted by organizer
        - β(Ē_all)^p: Complex contagion from ALL neighbors' mean energy
        - δ(1 - Ē_engaged): Decay based on ENGAGED neighbors only (mobilized + organizer)

        Returns:
            Dictionary with component contributions: outreach, contagion, decay
        """
        agent = self.agents[agent_id]

        # Contagion uses ALL neighbors' mean energy
        E_bar_all = self.get_neighbor_mean_energy(agent_id)

        # Decay uses only ENGAGED neighbors' mean energy
        E_bar_engaged = self.get_engaged_neighbor_mean_energy(agent_id)

        # Complex contagion: boost from neighborhood
        # Only applies when E_bar > 0 to avoid complex numbers with non-integer power
        if E_bar_all >= 0:
            contagion_term = self.beta * (E_bar_all**self.contagion_power)
        else:
            # Apply negative contagion: use absolute value for power, then negate
            contagion_term = -self.beta * (abs(E_bar_all) ** self.contagion_power)

        # Decay term based on engaged neighbors

        decay_term = self.delta * (1 - E_bar_engaged)

        # Update energy
        agent.energy = agent.energy + delta_outreach + contagion_term - decay_term

        agent.clip_energy()

        # Return components for tracking
        return {
            "outreach": delta_outreach,
            "contagion": contagion_term,
            "decay": -decay_term,  # Negative because decay removes energy
        }

    def check_persistence_gate(self, agent_id: int) -> bool:
        """
        Check if an agent passes the Persistence Gate to become an Organizer.

        An agent must:
        1. Have energy above omega (Ω)
        2. Be targeted by an existing organizer
        3. Maintain this for persistence_threshold consecutive ticks
        """
        agent = self.agents[agent_id]

        # Must be above threshold and targeted
        if agent.energy > self.omega and agent.is_targeted:
            agent.persistence_counter += 1

            if agent.persistence_counter >= self.persistence_threshold:
                return True
        else:
            # Reset counter if conditions not met
            agent.persistence_counter = 0

        return False

    def promote_to_organizer(self, agent_id: int) -> None:
        """
        Promote an agent to Organizer state and update clique connectivity.

        When an agent becomes an Organizer:
        1. Their state changes to ORGANIZER
        2. They form connections with other Organizers (building the clique)
        """
        agent = self.agents[agent_id]
        agent.state = AgentState.ORGANIZER
        agent.persistence_counter = 0
        agent.is_targeted = False
        agent.targeting_organizer = None

        # Add edges to other organizers (expand the clique)
        organizers = self.get_organizers()
        for org_id in organizers:
            if org_id != agent_id and not self.graph.has_edge(agent_id, org_id):
                self.graph.add_edge(agent_id, org_id)
                edge = (min(agent_id, org_id), max(agent_id, org_id))
                self.organizer_clique_edges.add(edge)

    def _store_snapshot(self) -> None:
        """
        Store a snapshot of the current network state for visualization.

        Captures agent energies, states, and network edges at this tick.
        """
        snapshot = {
            "tick": self.tick,
            "agents": {
                aid: {
                    "energy": agent.energy,
                    "state": agent.state.value,
                    "department_id": agent.department_id,
                    "is_targeted": agent.is_targeted,
                }
                for aid, agent in self.agents.items()
            },
            "edges": list(self.graph.edges()),
            "organizer_clique_edges": list(self.organizer_clique_edges),
        }
        self.snapshots.append(snapshot)

    def get_snapshot(self, tick: int) -> Optional[Dict]:
        """
        Get the network snapshot for a specific tick.

        Args:
            tick: The simulation tick to retrieve

        Returns:
            Snapshot dictionary or None if tick not found
        """
        for snapshot in self.snapshots:
            if snapshot["tick"] == tick:
                return snapshot
        return None

    def get_all_snapshots(self) -> List[Dict]:
        """Get all stored network snapshots."""
        return self.snapshots

    def get_organizers(self) -> List[int]:
        """Get list of all current organizers."""
        return [
            aid
            for aid, agent in self.agents.items()
            if agent.state == AgentState.ORGANIZER
        ]

    def get_mobilized(self) -> List[int]:
        """Get list of all mobilized agents."""
        return [
            aid
            for aid, agent in self.agents.items()
            if agent.state == AgentState.MOBILIZED
        ]

    def get_passive(self) -> List[int]:
        """Get list of all passive agents."""
        return [
            aid
            for aid, agent in self.agents.items()
            if agent.state == AgentState.PASSIVE
        ]

    def organizer_action(self, organizer_id: int) -> None:
        """
        Execute organizer action based on strategy parameter alpha.

        With probability alpha: perform 1-to-1 organizing (targeted)
        With probability (1-alpha): perform broadcast/mobilizing (diffuse)

        If stop_test is active, organizers take no actions.
        """
        if self.stop_test_active:
            return

        neighbors = list(self.graph.neighbors(organizer_id))

        if not neighbors:
            return

        # Decide strategy: 1-to-1 or broadcast
        if random.random() < self.alpha:
            # 1-to-1 Organizing: target specific non-organizer neighbor
            self._one_to_one_organizing(organizer_id, neighbors)
        else:
            # Broadcast/Mobilizing: small energy boost to all neighbors
            self._broadcast_mobilizing(organizer_id, neighbors)

    def _one_to_one_organizing(self, organizer_id: int, neighbors: List[int]) -> None:
        """
        Perform 1-to-1 organizing action.

        Target a non-organizer neighbor (preferring those not yet targeted)
        with a substantial energy boost.
        """
        targets = [n for n in neighbors if self.agents[n].state != AgentState.ORGANIZER]
        if not targets:
            return
        target_id = random.choice(targets)

        # Apply energy boost and mark as targeted
        target = self.agents[target_id]
        target.is_targeted = True
        target.targeting_organizer = organizer_id

        # Store delta for energy update phase
        if not hasattr(self, "_pending_energy_boosts"):
            self._pending_energy_boosts = {}

        self._pending_energy_boosts[target_id] = (
            self._pending_energy_boosts.get(target_id, 0.0) + self.outreach_energy
        )

    def _broadcast_mobilizing(self, organizer_id: int, neighbors: List[int]) -> None:
        """
        Perform broadcast mobilizing action.

        Randomly select a percentage of all nodes (simulating cold outreach
        like mass emails) and give them an energy boost

        The number of targets is determined by broadcast_reach_ratio parameter.
        """
        if not hasattr(self, "_pending_energy_boosts"):
            self._pending_energy_boosts = {}

        # Get all agents as potential targets
        all_agents = [aid for aid, agent in self.agents.items()]

        if not all_agents:
            return

        # Calculate number of targets based on broadcast_reach_ratio
        n_targets = max(1, int(len(self.agents) * self.broadcast_reach_ratio))
        n_targets = min(n_targets, len(all_agents))

        # Randomly select targets
        targets = random.sample(all_agents, n_targets)

        scaled_broadcast_energy = self.broadcast_energy

        for target_id in targets:
            self._pending_energy_boosts[target_id] = (
                self._pending_energy_boosts.get(target_id, 0.0)
                + scaled_broadcast_energy
            )

    def update_agent_states(self) -> None:
        """
        Update agent states based on energy levels.

        Passive → Mobilized: when energy > mobilizing_threshold
        Mobilized → Organizer: when passing Persistence Gate
        Mobilized → Passive: when energy ≤ mobilizing_threshold
        Organizer → Mobilized: when below omega for persistence_threshold consecutive ticks
        """
        for agent_id, agent in self.agents.items():
            if agent.state == AgentState.ORGANIZER:
                # Organizers require persistence gate to demote (same as promotion)
                if agent.energy <= self.omega:
                    agent.demotion_counter += 1
                    # Only demote after being below threshold for persistence_threshold ticks
                    if agent.demotion_counter >= self.persistence_threshold:
                        if agent.energy <= self.mobilizing_threshold:
                            agent.state = AgentState.PASSIVE
                        elif agent.energy <= self.omega:
                            agent.state = AgentState.MOBILIZED
                        agent.persistence_counter = 0
                        agent.demotion_counter = 0
                        # Note: We keep the clique edges as they represent established relationships
                else:
                    # Reset counter if energy goes back above omega
                    agent.demotion_counter = 0
                continue

            if agent.energy > self.omega:
                # Check for promotion to Organizer
                if agent.is_targeted and self.check_persistence_gate(agent_id):
                    self.promote_to_organizer(agent_id)
                elif agent.state == AgentState.PASSIVE:
                    agent.state = AgentState.MOBILIZED
            elif agent.energy > self.mobilizing_threshold:
                if agent.state == AgentState.PASSIVE:
                    agent.state = AgentState.MOBILIZED
            else:
                # Energy ≤ mobilizing_threshold: demote to passive
                if agent.state == AgentState.MOBILIZED:
                    agent.state = AgentState.PASSIVE
                    agent.persistence_counter = 0

            # Clip energy: non-organizers cannot exceed omega threshold perform after update

    def step(self) -> Dict:
        """
        Execute one simulation tick.

        Returns a dictionary of metrics for this tick.
        """
        self.tick += 1

        # Initialize pending energy boosts
        self._pending_energy_boosts = {}

        # Phase 1: Organizers take actions
        organizers = self.get_organizers()
        for org_id in organizers:
            self.organizer_action(org_id)

        # Phase 2: Update all agent energies and track components
        self.energy_components = {"outreach": 0.0, "contagion": 0.0, "decay": 0.0}
        for agent_id in self.agents:
            delta_e = self._pending_energy_boosts.get(agent_id, 0.0)
            components = self.update_energy(agent_id, delta_e)
            # Accumulate components
            self.energy_components["outreach"] += components["outreach"]
            self.energy_components["contagion"] += components["contagion"]
            self.energy_components["decay"] += components["decay"]

        # Phase 3: Update agent states
        self.update_agent_states()

        # Clip energies according to ideological ranges and organizer max threshold
        for agent_id in self.agents:
            self.agents[agent_id].clip_energy(
                max_threshold=(
                    None
                    if self.agents[agent_id].state == AgentState.ORGANIZER
                    else self.omega
                )
            )

        # Phase 4: Record history
        for agent in self.agents.values():
            agent.update_history()

        # Phase 5: Store snapshot for visualization
        self._store_snapshot()

        # Clean up
        self._pending_energy_boosts = {}

        # Return metrics for this tick
        return self.get_metrics()

    def run(self, n_ticks: int, stop_test_tick: Optional[int] = None) -> List[Dict]:
        """
        Run the simulation for n_ticks.

        Args:
            n_ticks: Number of simulation steps to run
            stop_test_tick: If provided, activate stop test at this tick

        Returns:
            List of metric dictionaries, one per tick
        """
        self.stop_test_tick = stop_test_tick
        metrics_history = []

        for t in range(n_ticks):
            # Check for stop test activation
            if stop_test_tick is not None and self.tick >= stop_test_tick:
                self.stop_test_active = True

            metrics = self.step()
            metrics_history.append(metrics)

        return metrics_history

    def activate_stop_test(self) -> None:
        """Manually activate the stop test."""
        self.stop_test_active = True
        self.stop_test_tick = self.tick

    def deactivate_stop_test(self) -> None:
        """Deactivate the stop test."""
        self.stop_test_active = False

    def get_metrics(self) -> Dict:
        """
        Calculate and return all simulation metrics.

        Returns dictionary containing:
        - tick: Current simulation tick
        - total_energy: Sum of all agent energies
        - mean_energy: Mean agent energy
        - mobilized_count: Number of mobilized agents
        - organizer_count: Number of organizers
        - passive_count: Number of passive agents
        - dsi: Department Spread Index
        - lwr: Leader-Worker Ratio
        - clustering_coefficient: Network clustering coefficient
        - global_efficiency: Network global efficiency
        - reach: Fraction of non-passive agents
        """
        energies = [agent.energy for agent in self.agents.values()]

        organizers = self.get_organizers()
        mobilized = self.get_mobilized()
        passive = self.get_passive()

        n_total = len(self.agents)
        current_total_energy = sum(energies)

        # Calculate energy gradient (change from previous tick)
        if self.previous_total_energy is not None:
            energy_gradient = current_total_energy - self.previous_total_energy
        else:
            energy_gradient = 0.0

        # Update previous energy for next tick
        self.previous_total_energy = current_total_energy

        return {
            "tick": self.tick,
            "total_energy": current_total_energy,
            "mean_energy": np.mean(energies),
            "energy_gradient": energy_gradient,
            "energy_from_outreach": self.energy_components["outreach"],
            "energy_from_contagion": self.energy_components["contagion"],
            "energy_from_decay": self.energy_components["decay"],
            "mobilized_count": len(mobilized),
            "organizer_count": len(organizers),
            "passive_count": len(passive),
            "dsi": self._calculate_dsi(),
            "lwr": self._calculate_lwr(),
            "clustering_coefficient": nx.average_clustering(self.graph),
            "global_efficiency": nx.global_efficiency(self.graph),
            "reach": (len(organizers) + len(mobilized)) / n_total if n_total > 0 else 0,
            "stop_test_active": self.stop_test_active,
        }

    def _calculate_dsi(self) -> float:
        """
        Calculate Department Spread Index (DSI).

        DSI measures how evenly organizers are distributed across departments.
        DSI = 1 - (std of organizer counts per dept / max possible std)

        Returns 1.0 if organizers are perfectly spread, 0.0 if all in one dept.
        """
        organizers = self.get_organizers()
        n_org = len(organizers)

        if n_org == 0:
            return 0.0

        # Count organizers per department
        dept_counts = [0] * self.n_departments
        for org_id in organizers:
            dept = self.agents[org_id].department_id
            dept_counts[dept] += 1

        # Calculate normalized spread
        # Perfect spread: equal organizers in each department
        # Worst spread: all organizers in one department
        std = np.std(dept_counts)
        max_std = np.sqrt((n_org**2) * (self.n_departments - 1) / self.n_departments)

        if max_std == 0:
            return 1.0

        return 1.0 - (std / max_std)

    def _calculate_lwr(self) -> float:
        """
        Calculate Leader-Worker Ratio (LWR).

        LWR = number of organizers / number of mobilized (engaged) workers

        Higher LWR indicates more "leadership density" - closer to union
        organizing model where many leaders emerge from the base.
        """
        organizers = self.get_organizers()
        mobilized = self.get_mobilized()

        n_workers = len(mobilized)  # Workers = engaged non-organizers
        n_leaders = len(organizers)

        if n_workers == 0:
            return float("inf") if n_leaders > 0 else 0.0

        return n_leaders / n_workers

    def get_state_dict(self) -> Dict:
        """
        Get complete simulation state as a dictionary.

        Used for JSON serialization and state persistence.
        """
        return {
            "parameters": {
                "n_departments": self.n_departments,
                "agents_per_dept": self.agents_per_dept,
                "p_in": self.p_in,
                "p_out": self.p_out,
                "seed_size": self.seed_size,
                "alpha": self.alpha,
                "beta": self.beta,
                "delta": self.delta,
                "omega": self.omega,
                "mobilizing_threshold": self.mobilizing_threshold,
                "persistence_threshold": self.persistence_threshold,
                "outreach_energy": self.outreach_energy,
                "broadcast_energy": self.broadcast_energy,
                "contagion_power": self.contagion_power,
                "organizer_potential_ratio": self.organizer_potential_ratio,
                "seed": self.seed,
            },
            "state": {
                "tick": self.tick,
                "stop_test_active": self.stop_test_active,
                "stop_test_tick": self.stop_test_tick,
            },
            "agents": {
                aid: {
                    "energy": agent.energy,
                    "state": agent.state.value,
                    "department_id": agent.department_id,
                    "persistence_counter": agent.persistence_counter,
                    "is_targeted": agent.is_targeted,
                    "targeting_organizer": agent.targeting_organizer,
                    "energy_min": agent.energy_min,
                    "energy_max": agent.energy_max,
                }
                for aid, agent in self.agents.items()
            },
            "network": {
                "edges": list(self.graph.edges()),
                "organizer_clique_edges": list(self.organizer_clique_edges),
            },
            "seed_clique": list(self.seed_clique),
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict) -> "UnionSim":
        """
        Reconstruct simulation from a state dictionary.

        Used for loading saved simulation states.
        """
        params = state_dict["parameters"]

        # Create new simulation with same parameters
        sim = cls(**params)

        # Restore simulation state
        sim.tick = state_dict["state"]["tick"]
        sim.stop_test_active = state_dict["state"]["stop_test_active"]
        sim.stop_test_tick = state_dict["state"]["stop_test_tick"]

        # Restore agent states
        for aid_str, agent_data in state_dict["agents"].items():
            aid = int(aid_str)
            if aid in sim.agents:
                sim.agents[aid].energy = agent_data["energy"]
                sim.agents[aid].state = AgentState(agent_data["state"])
                sim.agents[aid].persistence_counter = agent_data["persistence_counter"]
                sim.agents[aid].is_targeted = agent_data["is_targeted"]
                sim.agents[aid].targeting_organizer = agent_data["targeting_organizer"]
                # Restore ideological range if present
                if "energy_min" in agent_data:
                    sim.agents[aid].energy_min = agent_data["energy_min"]
                if "energy_max" in agent_data:
                    sim.agents[aid].energy_max = agent_data["energy_max"]

        # Restore network edges
        sim.graph.clear_edges()
        for edge in state_dict["network"]["edges"]:
            sim.graph.add_edge(edge[0], edge[1])

        sim.organizer_clique_edges = set(
            tuple(e) for e in state_dict["network"]["organizer_clique_edges"]
        )
        sim.seed_clique = set(state_dict["seed_clique"])

        return sim
