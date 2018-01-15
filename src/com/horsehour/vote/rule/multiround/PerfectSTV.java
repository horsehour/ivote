package com.horsehour.vote.rule.multiround;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;

/**
 * Suppose we are given a perfect predictor, who can always tell what the
 * winners from a state. Therefore, we could detect the optimal searching path
 * from top-to-bottom for all winners with the minimum cost (e.g. minimum number
 * of nodes to extend), and as soon as possible. The goal to collect all winners
 * with minimum cost as soon as possible is called the early discovery strategy.
 * Based on the optimal early discovery strategy, we define the metric to
 * measure how well a searching strategy. Once we know how well a searching path
 * approximate to the optimal one, the immediate next step is to construct a
 * pool of searching strategies, and sampling from it we can implement the
 * reinforcement learning algorithm for an optimal searching policy. On the
 * other hand, the results at least can indicate the potential gap between the
 * searching strategy from our heuristic method and the optimal solution.
 * 
 * @author Chunheng Jiang
 * @since Apr 5, 2017
 * @version 1.0
 */
public class PerfectSTV {
	public Node root;
	public List<Node> leaves;

	public int numItem, numNode, numWinner;

	public Profile<Integer> profile;
	public Map<Integer, Integer> trace;
	public List<Integer> winners;
	public Map<List<Integer>, Node> visited;

	public String base = "/users/chjiang/github/csc/";
	public String dataset = "soc-3";

	public int rank = 0;

	public PerfectSTV() {}

	public class Node {
		public int order = 0;
		public int depth = 0;
		public float priority = -1;

		public boolean leaf = false;
		public double q = 0;

		public List<Integer> state;

		public List<Node> children;
		public List<Node> parents;
		public List<Integer> winners;

		public Node(List<Integer> candidates) {
			this.state = new ArrayList<>(candidates);

			this.winners = new ArrayList<>(1);
			this.parents = new ArrayList<>(1);
			this.children = new ArrayList<>(1);

			if (state.size() == 1)
				this.leaf = true;
		}

		public Node(List<Integer> candidates, Integer eliminated) {
			this.state = new ArrayList<>(candidates);
			this.state.remove(eliminated);

			this.winners = new ArrayList<>(1);
			this.parents = new ArrayList<>(1);
			this.children = new ArrayList<>(1);

			if (state.size() == 1)
				this.leaf = true;
		}
	}

	/**
	 * depth first search
	 */
	void dfs() {
		LinkedList<Node> fringe = new LinkedList<>();
		fringe.add(root);

		Node node = null;
		while (!fringe.isEmpty()) {
			node = fringe.pollLast();
			List<Integer> state = node.state;

			Node previous = visited.get(state);
			if (previous != null) { // visited state
				previous.parents.addAll(node.parents);
				// destroy link: node a & its parent(s) P
				// rebuild link: visited node b & P
				for (Node parent : node.parents) {
					parent.children.remove(node);
					parent.children.add(previous);
				}
				continue;
			} else if (state.size() == 1) {
				node.winners.add(state.get(0));
				leaves.add(node); // collect all leaves
				visited.put(state, node);
				trace.put(++numWinner, ++numNode);
				continue;
			}

			int[] scores = scoring(profile, state);
			int[] min = MathLib.argmin(scores);
			for (int i : min) {
				Node child = new Node(state, state.get(i));
				child.depth = node.depth + 1;
				child.parents.add(node);
				node.children.add(child);
			}
			fringe.addAll(node.children);
			visited.put(state, node);// label current state as visited
			numNode++;
		}
	}

	/**
	 * Calculate plurality score from a preference profile with those unstated
	 * candidates eliminated
	 * 
	 * @param profile
	 * @param state
	 * @return Plurality score of these candidates
	 */
	public int[] scoring(Profile<Integer> profile, List<Integer> state) {
		int[] votes = profile.votes;
		int[] scores = new int[state.size()];
		int c = 0;
		for (Integer[] pref : profile.data) {
			for (int i = 0; i < pref.length; i++) {
				int item = pref[i];
				int index = state.indexOf(item);
				/** item has been eliminated **/
				if (index == -1)
					continue;
				scores[index] += votes[c];
				break;
			}
			c++;
		}
		return scores;
	}

	/**
	 * Remove candidates who have never been the first choice of any voter and
	 * reconstruct the profile
	 * 
	 * @param profile
	 * @param state
	 * @return reconstructed profile if there are any candidate has being
	 *         eliminated, otherwise return the original profile
	 */
	public Profile<Integer> preprocess(Profile<Integer> profile, List<Integer> state) {
		int[] scores = scoring(profile, state);
		List<Integer> eliminated = new ArrayList<>();
		for (int i = 0; i < scores.length; i++) {
			if (scores[i] == 0)
				eliminated.add(i);
		}

		if (!eliminated.isEmpty()) {
			state.removeAll(eliminated);
			return profile.reconstruct(state);
		} else
			return profile;
	}

	/**
	 * @param profile
	 * @return All possible winners with various tie-breaking rules
	 */
	public List<Integer> getAllWinners(Profile<Integer> profile) {
		this.numNode = 0;
		this.numWinner = 0;

		this.visited = new HashMap<>();

		Integer[] items = profile.getSortedItems();
		List<Integer> state = new ArrayList<>(Arrays.asList(items));
		this.profile = preprocess(profile, state);
		this.numItem = state.size();
		this.root = new Node(state);
		this.leaves = new ArrayList<>();

		this.trace = new HashMap<>();
		this.dfs();

		List<Integer> winners = new ArrayList<>();
		for (Node node : leaves)
			winners.addAll(node.winners);
		Collections.sort(winners);
		return winners;
	}

	/**
	 * Propagate winners from leaves to the root node
	 */
	void upwardPropagate(List<Node> nodes) {
		Set<Node> parentList = new HashSet<>();

		for (Node node : nodes) {
			List<Node> parents = node.parents;
			if (parents.isEmpty())
				return;

			List<Integer> winnerList = node.winners;
			for (Node parent : parents) {
				if (parent.winners.isEmpty()) {
					parent.winners = new ArrayList<>(winnerList);
				} else {
					// deliver children's winners to parents
					for (int winner : winnerList) {
						if (!parent.winners.contains(winner))
							parent.winners.add(winner);
					}
				}
			}
			parentList.addAll(parents);
		}
		upwardPropagate(new ArrayList<>(parentList));
	}

	/**
	 * Search the (sub)-optimal episode in top-down pattern
	 */
	public void getSubOptimalEpisode() {
		this.upwardPropagate(leaves);

		this.numNode = 0;
		this.numWinner = 0;

		this.visited = new HashMap<>();
		this.trace = new HashMap<>();
		this.winners = new ArrayList<>();

		this.rank = 1;

		LinkedList<Node> fringe = new LinkedList<>();
		fringe.add(root);
		root.order = rank;

		Node next = null;
		while (!fringe.isEmpty()) {
			next = fringe.pollLast();

			List<Integer> state = next.state;
			Node previous = visited.get(state);

			if (previous != null) {// visited state
				continue;
			}

			next.order = rank++;
			if (state.size() == 1) {
				visited.put(state, next);
				numNode++;
				numWinner++;
				winners.add(state.get(0));
				trace.put(numWinner, numNode);
				continue;
			}

			if (canPrune(winners, state)) {
				visited.put(state, next);
				numNode++;
				continue;
			}

			shuffle(next);

			fringe.addAll(next.children);

			visited.put(state, next);
			numNode++;
		}
	}

	/**
	 * shuffle children to keep the strongly connected ones with most fresh
	 * winners and drop away the other redundant ones
	 * 
	 * @param next
	 */
	void shuffle(Node next) {
		int numChild = next.children.size();
		if (numChild == 1)
			return;

		List<Integer> ind = new ArrayList<>();
		for (int i = 0; i < numChild; i++) {
			ind.add(i);
			next.children.get(i).priority = -1;
		}

		Set<Integer> set = new HashSet<>();
		set.addAll(winners);

		int indBest = 0, maxFresh = -1;
		while (maxFresh != 0) {
			maxFresh = 0;
			for (int i : ind) {
				int fresh = 0;
				Node child = next.children.get(i);
				for (int winner : child.winners) {
					if (!set.contains(winner))
						fresh++;
				}

				if (fresh > maxFresh) {
					maxFresh = fresh;
					indBest = i;
				} else if (fresh == maxFresh) {
					/** multiple - randomly select one **/
					// TODO: undetermined
					double rnd = Math.random();
					if (rnd > 0.5)
						indBest = i;
				}
			}

			if (maxFresh > 0 && !ind.isEmpty()) {
				ind.remove(new Integer(indBest));
				// Adding children to the queue based on their priorities
				// next.children.get(indBest).priority = ind.size() * maxFresh;
				next.children.get(indBest).priority = maxFresh;
				set.addAll(next.children.get(indBest).winners);
			}
		}

		/** eliminate all redundant branches **/
		for (int i = ind.size() - 1; i >= 0; i--) {
			Node child = next.children.get(ind.get(i));
			next.children.remove(child);
		}

		/** sort based on priority **/
		next.children.sort((nd1, nd2) -> Double.compare(nd1.priority, nd2.priority));
	}

	/**
	 * shuffle children randomly and picked subset of them covering the winners
	 * of current node
	 * 
	 * @param next
	 */
	void shuffleRND(Node next) {
		int numChild = next.children.size();
		if (numChild == 1)
			return;

		List<Integer> ind = new ArrayList<>();
		for (int i = 0; i < numChild; i++)
			ind.add(i);
		Collections.shuffle(ind);

		Set<Integer> set = new HashSet<>();
		set.addAll(winners);

		List<Integer> rmvList = new ArrayList<>();
		int fresh = 1, c = ind.size();
		for (int i : ind) {
			Node child = next.children.get(i);
			fresh = computeFresh(child, set);
			if (fresh > 0) {
				next.children.get(i).priority = c;
				set.addAll(next.children.get(i).winners);
				c -= 1;
			} else {
				rmvList.add(i);
			}
		}

		Collections.sort(rmvList);
		/** eliminate all redundant branches **/
		for (int i = rmvList.size() - 1; i >= 0; i--) {
			Node child = next.children.get(rmvList.get(i));
			next.children.remove(child);
		}

		/** sort based on priority **/
		next.children.sort((nd1, nd2) -> Double.compare(nd1.priority, nd2.priority));
	}

	/**
	 * Count the number of alternatives who are never elected in previous rounds
	 * 
	 * @param child
	 * @param winners
	 * @return freshness of a nodes
	 */
	int computeFresh(Node child, Set<Integer> winners) {
		int fresh = 0;
		for (int winner : child.winners)
			if (!winners.contains(winner))
				fresh++;
		return fresh;
	}

	/**
	 * Evaluate whether the remaining candidates in current state are all have
	 * been elected as winners in previous steps
	 * 
	 * @param winners
	 * @param state
	 * @return true if the state is a subset of the known winners; false
	 *         otherwise
	 */
	boolean canPrune(List<Integer> winners, List<Integer> state) {
		if (winners.isEmpty())
			return false;

		boolean prune = true;
		for (int item : state) {
			if (!winners.contains(item)) {
				prune = false;
				break;
			}
		}
		return prune;
	}

	/***
	 * Collect specific number of episodes by repeating the experiments
	 * 
	 * @param nTrials
	 * @return list of episodes
	 */
	public List<List<Node>> getEpisodes(int nTrials) {
		upwardPropagate(leaves);
		List<List<Node>> episodes = new ArrayList<>();
		episodes.add(getEpisode());
		return episodes;
	}

	/**
	 * Sample a route for the multi-round STV and use the episode to train the
	 * reinforcement learning model
	 * 
	 * @return a route to search all winners
	 */
	public List<Node> getEpisode() {
		this.numNode = 0;
		this.numWinner = 0;

		this.visited = new HashMap<>();
		this.trace = new HashMap<>();
		this.winners = new ArrayList<>();

		List<Node> episode = new ArrayList<>();

		this.rank = 1;

		LinkedList<Node> fringe = new LinkedList<>();
		fringe.add(root);
		root.order = rank;

		Node next = null;
		while (!fringe.isEmpty()) {
			next = fringe.pollLast();

			List<Integer> state = next.state;
			Node previous = visited.get(state);

			if (previous != null) {// visited state
				continue;
			}

			next.order = rank++;
			if (state.size() == 1) {
				episode.add(next);
				visited.put(state, next);
				numNode++;
				numWinner++;
				winners.add(state.get(0));
				trace.put(numWinner, numNode);
				continue;
			}

			if (canPrune(winners, state)) {
				episode.add(next);
				visited.put(state, next);
				numNode++;
				continue;
			}

			double rnd = Math.random();
			if (rnd < 0.5)
				shuffleRND(next);
			else
				shuffle(next);

			fringe.addAll(next.children);

			episode.add(next);
			visited.put(state, next);
			numNode++;
		}
		return episode;
	}

	Function<Integer, String> space = d -> {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < d; i++)
			sb.append("  ");
		return sb.toString();
	};

	public String getVotingTree(Node node) {
		String name = node.order + " : " + node.state.toString().replaceAll("\\[|\\]| ", "");
		if (node.state.size() > 1 && node.winners != null)
			name += node.winners;
		name = "\"name\": \"" + name + "\"";

		StringBuffer sb = new StringBuffer();
		if (node.children.isEmpty()) {
			sb.append(space.apply(node.depth)).append("{").append(name).append("}");
		} else {
			int nc = node.children.size();
			sb.append(space.apply(node.depth)).append("{\n");
			sb.append(space.apply(node.depth + 1)).append(name).append(", \n");
			sb.append(space.apply(node.depth + 1)).append("\"children\": [\n");
			for (int i = 0; i < nc; i++) {
				Node child = node.children.get(i);
				sb.append(getVotingTree(child));
				if (i < nc - 1)
					sb.append(",");
				sb.append("\n");
			}
			sb.append(space.apply(node.depth + 1)).append("]\n");
			sb.append(space.apply(node.depth)).append("}");
		}
		return sb.toString();
	}

	/**
	 * The k possible intervals are: [0,1/k], [1/k, 2/k],...,[(k-1)/k, k/k].
	 * Given a percentage a > 0, ceil(k*a) - 1 should be the index of the
	 * interval starting from 0 that the percentage fails into
	 */
	static BiFunction<Integer, Map<Integer, Integer>, int[]> transform = (k, trace) -> {
		int[] intervals = new int[k];
		int n = trace.size();
		for (int i : trace.keySet()) {
			int ind = (int) Math.ceil((k * i * 1.0 / n)) - 1;
			intervals[ind] += trace.get(i);
		}
		return intervals;
	};

	public static void main1(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/users/chjiang/github/csc/";
		String dataset = "soc-3-hardcase";

		int k = 10;

		OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };

		boolean heuristic = false, cache = true, pruning = true;
		boolean sampling = false, recursive = false;
		int pFunction = 0;
		STVPlus2 rule = new STVPlus2(heuristic, cache, pruning, sampling, recursive, pFunction);

		PerfectSTV perfect = new PerfectSTV();

		float[] avg = new float[k];
		AtomicInteger count = new AtomicInteger(0);
		Files.list(Paths.get(base + dataset)).forEach(input -> {
			String name = input.toFile().getName();
			int m = Integer.parseInt(name.substring(1, name.indexOf("N")));
			if (m > 30)
				return;

			Profile<Integer> profile = DataEngine.loadProfile(input);

			StringBuffer sb = new StringBuffer();

			List<Integer> winners = rule.getAllWinners(profile);
			if (winners.size() == 1)
				return;

			int[] baseline = transform.apply(k, rule.trace);

			sb.append(name + "\t" + perfect.numNode + "\t" + rule.trace);

			perfect.getAllWinners(profile);
			perfect.getSubOptimalEpisode();

			int[] proposed = transform.apply(k, perfect.trace);

			sb.append("\t" + perfect.numNode + "\t" + perfect.trace);

			float[] improved = new float[k];
			for (int i = 0; i < k; i++) {
				avg[i] += improved[i];
				if (baseline[i] > 0)
					improved[i] = (baseline[i] - proposed[i]) * 1.0F / baseline[i];
			}
			sb.append("\t").append(Arrays.toString(improved)).append("\n");

			try {
				Files.write(Paths.get(base + "early.csv"), sb.toString().getBytes(), options);
			} catch (IOException e) {
				e.printStackTrace();
			}
			System.out.println(name);
			count.getAndIncrement();
		});

		for (int i = 0; i < k; i++)
			avg[i] /= count.get();
		System.out.println(Arrays.toString(avg));

		TickClock.stopTick();
	}

	public static void main3(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/users/chjiang/github/csc/";

		Path input = Paths.get(base + "early.csv");
		Path output = Paths.get(base + "early.perf.csv");

		StringBuffer sb = new StringBuffer();
		for (String line : Files.readAllLines(input)) {
			int ind = line.lastIndexOf("[");
			String perf = line.substring(ind + 1, line.length() - 1);
			perf = perf.replaceAll(" ", "").replaceAll(",", "\t");
			sb.append(perf).append("\n");
		}

		Files.write(output, sb.toString().getBytes());
		TickClock.stopTick();
	}

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/users/chjiang/github/csc/";
		String dataset = "soc-3-stv";
		// String dataset = "soc-5-stv";
		// String dataset = "stv-m30n30-7000";

		Path input = Paths.get(base + dataset + "/M20N10-150.csv");
		Profile<Integer> profile = DataEngine.loadProfile(input);

		PerfectSTV rule = new PerfectSTV();
		System.out.println(rule.getAllWinners(profile));
		System.out.println(rule.trace);
		System.out.println(rule.numNode);

		int signal = 0;
		if (signal == 0)
			rule.getSubOptimalEpisode();
		else {
			rule.upwardPropagate(rule.leaves);
			rule.getEpisode();
		}

		// rule.getEpisodes(10);
		System.out.println(rule.trace);

		String tree = rule.getVotingTree(rule.root);
		Files.write(Paths.get(base + "/votetree/stv.json"), tree.getBytes());

		TickClock.stopTick();
	}

	public static void main1111(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/users/chjiang/github/csc/";
		String dataset = "soc-6-stv";

		PerfectSTV rule = new PerfectSTV();
		OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };
		Path output = Paths.get("/users/chjiang/github/pycharm/stv/perf/ed.m5.best.txt");
		StringBuffer sb = null;
		for (int n = 3; n <= 10; n++) {
			for (int k = 1; k <= 5000; k++) {
				String name = "M5N" + n + "-" + k + ".csv";
				Path input = Paths.get(base + dataset + "/" + name);
				Profile<Integer> profile = DataEngine.loadProfile(input);
				rule.getAllWinners(profile);
				// System.out.println(rule.trace);
				rule.getSubOptimalEpisode();
				sb = new StringBuffer().append(dataset).append("\t").append(name);
				sb.append("\t").append(rule.trace).append("\n");
				Files.write(output, sb.toString().getBytes(), options);
				System.out.println(name);
			}
		}

		TickClock.stopTick();
	}

	public static void main323(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/users/chjiang/github/csc/";
		String dataset = "soc-3-hardcase";

		PerfectSTV rule = new PerfectSTV();

		boolean heuristic = false, cache = true, pruning = true;
		boolean sampling = false, recursive = false;
		int pFunction = 0;
		STVPlus2 stv = new STVPlus2(heuristic, cache, pruning, sampling, recursive, pFunction);

		Path output1 = Paths.get("/users/chjiang/github/csc/ed.perfect.txt");
		Path output2 = Paths.get("/users/chjiang/github/csc/ed.stv.txt");

		OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };

		Files.list(Paths.get(base + dataset)).forEach(input -> {
			StringBuffer sb1 = new StringBuffer(), sb2 = new StringBuffer();

			String name = input.toFile().getName();
			int ind = name.indexOf("N");
			int m = Integer.parseInt(name.substring(1, ind));
			if (m > 30)
				return;

			Profile<Integer> profile = DataEngine.loadProfile(input);
			rule.getAllWinners(profile);
			sb1.append(name).append("\t");
			sb1.append(rule.trace.values()).append("\n");
			stv.getAllWinners(profile);
			sb2.append(name).append("\t");
			sb2.append(stv.trace.values()).append("\n");

			try {
				Files.write(output1, sb1.toString().getBytes(), options);
				Files.write(output2, sb2.toString().getBytes(), options);
			} catch (IOException e) {
				e.printStackTrace();
			}
		});

		TickClock.stopTick();
	}
}