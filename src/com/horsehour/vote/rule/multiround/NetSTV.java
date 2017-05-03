package com.horsehour.vote.rule.multiround;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;
import com.horsehour.vote.data.FeatureLab;

/**
 * Network-based STV with Reinforcement learning
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 3:06:45 AM, Jan 10, 2017
 */

public class NetSTV {
	public String dataset = "soc-3";
	public String base = "/users/chjiang/github/csc/";

	public int numItemTotal;
	public int numWinner, numNode;
	public Map<Integer, Integer> trace;

	public List<Double> rewards;
	public int nSample = 0, maxSample = 30000;

	public Node root;

	public LinkedList<Node> nodes = new LinkedList<>();

	public Map<Integer, Integer> freq;

	public Profile<Integer> profile;

	public Set<List<Integer>> visited;

	public double alpha = 0.1, gamma = 0.8;

	// rewarding unit
	public double unit = 0;

	public double[] qFunction;
	public double[] features;
	public Node optimal;

	public List<List<Node>> network;

	public NetSTV() {}

	public class Node {
		public int depth = 0;
		public boolean leaf = false;

		public double q = 0;

		public List<Integer> state;

		public List<Node> children;
		public List<Node> parents;

		public Node(List<Integer> candidates) {
			this.state = new ArrayList<>(candidates);
			if (state.size() == 1)
				leaf = true;
		}

		public Node(List<Integer> candidates, Integer eliminated) {
			this.state = new ArrayList<>(candidates);
			this.state.remove(eliminated);
			if (state.size() == 1)
				leaf = true;
		}
	}

	/**
	 * Extend the node to produce its children if possible
	 * 
	 * @param node
	 */
	void extend(Node node) {
		if (node.children == null || node.children.isEmpty()) {
			List<Integer> state = node.state;
			int[] scores = scoring(profile, state);
			node.children = new ArrayList<>();
			int[] min = MathLib.argmin(scores);
			for (int i = 0; i < min.length; i++) {
				Node child = new Node(state, state.get(min[i]));
				nodes.add(node);
				child.depth = node.depth + 1;
				node.children.add(child);
			}
		}
	}

	/**
	 * Evaluate the current state with one candidate
	 * 
	 * @param node
	 * @return true if the current state contains only one candidate
	 */
	int isLeaf(Node node) {
		List<Integer> state = node.state;
		int signal = 0;
		if (node.leaf) {
			int item = state.get(0);
			int count = freq.get(item);
			if (count == 0) {
				numNode++;
				numWinner++;
				visited.add(state);
				trace.put(numWinner, numNode);
				signal = 1;
			} else
				signal = 2;
			freq.put(item, count + 1);
		}
		return signal;
	}

	/**
	 * Evaluate whether the remaining candidates in current state are all have
	 * been elected as winners in previous steps
	 * 
	 * @param state
	 * @return true if the state is a subset of the known winners; false
	 *         otherwise
	 */
	boolean canPrune(List<Integer> state) {
		for (int item : state) {
			// some candidates is not in the known-winner set
			if (freq.get(item) == 0) {
				return false;
			}
		}
		visited.add(state);
		return true;
	}

	/**
	 * Evaluate whether the state has been visited
	 * 
	 * @param state
	 * @return true if the state has been visited, false otherwise
	 */
	boolean hasVisited(List<Integer> state) {
		if (visited.contains(state)) {
			return true;
		} else {
			return false;
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

		if (eliminated.isEmpty())
			return profile;
		else {
			state.removeAll(eliminated);
			return profile.reconstruct(state);
		}
	}

	/**
	 * <p>
	 * TODO: Reward is a little bit smaller. <br/>
	 * TODO: Before the agent collected all winners, all paths with a state
	 * being visited for twice should be discouraged. <br/>
	 * </p>
	 * Rewarding function is defined based on many attributes to encourage the
	 * agent searches the way as expected
	 * 
	 * @param node
	 * @return reward
	 */
	public double getReward(Node node) {
		double reward = -1 * unit;
		List<Integer> state = node.state;
		int signal = isLeaf(node);
		if (signal == 1)
			reward += 10 * unit;
		else if (signal == 2)
			reward -= 2 * unit / freq.get(state.get(0));
		else {
			boolean flag = hasVisited(state);
			if (flag || canPrune(state))
				reward -= 5 * unit;
			else if (!flag) {
				visited.add(node.state);
			}
		}
		return reward;
	}

	/***
	 * Compute the Q-value
	 * 
	 * @param node
	 * @return q value
	 */
	public double getQ(Node node) {
		List<Integer> state = node.state;
		features = FeatureLab.getF1(profile, state, numItemTotal);
		double value = MathLib.Matrix.dotProd(qFunction, features);
		return value;
	}

	/***
	 * Update the Q-function
	 * 
	 * @param node
	 * @param next
	 */
	public void update(Node node, Node next) {
		double diff = getReward(node) + gamma * next.q - node.q;
		qFunction = MathLib.Matrix.lin(qFunction, 1.0, features, alpha * diff);
		// sum normalization
		MathLib.Scale.sum(qFunction);
	}

	/**
	 * Compute optimal step and collect all non-optimal nodes into queue
	 * 
	 * @param node
	 */
	public void computeOptimalStep(Node node) {
		double maxQ = Double.MIN_VALUE;
		int maxInd = 0, i = 0;
		for (Node child : node.children) {
			child.q = getQ(child);
			if (maxQ < child.q) {
				maxQ = child.q;
				maxInd = i;
			}
			i += 1;
		}
		optimal = node.children.get(maxInd);
	}

	Comparator<Node> comparator = (n1, n2) -> Double.compare(n1.q, n2.q);

	/**
	 * Restart the training with another episode as the input
	 * 
	 * @return start node
	 */
	public Node restart() {
		// for (Node node : nodes)
		// node.q = getQ(node);
		// nodes.sort(comparator);
		// return nodes.getLast();
		int ind = MathLib.Rand.sample(0, nodes.size());
		return nodes.get(ind);
	}

	/**
	 * Q-Learning starting from one specific state
	 * 
	 * @param node
	 */
	public void qLearn(Node node) {
		while (node != null) {
			if (isLeaf(node) == 0) {
				extend(node);
				computeOptimalStep(node);
				update(node, optimal);
				node = optimal;
			} else {
				rewards.add(getReward(node));
				nSample++;
				// System.out.println(++nSample + " : " + getReward(node));
				if (nSample < maxSample)
					node = restart();
				else
					break;
			}
		}
	}

	/**
	 * Q-Learning
	 * 
	 * @param profile
	 */
	public void qLearn(Profile<Integer> profile) {
		this.visited = new HashSet<>();
		this.freq = new HashMap<>();
		for (int i = 0; i < numItemTotal; i++)
			freq.put(i, 0);

		this.trace = new HashMap<>();
		this.rewards = new ArrayList<>();

		Integer[] items = profile.getSortedItems();
		List<Integer> state = new ArrayList<>(Arrays.asList(items));
		this.profile = preprocess(profile, state);
		this.network = new ArrayList<>();
		this.root = new Node(state);
		this.network.add(Arrays.asList(root));

		features = FeatureLab.getF1(profile, state, numItemTotal);
		if (qFunction == null)
			qFunction = new double[features.length];
		MathLib.Rand.distribution(qFunction);

		this.unit = numItemTotal * 1.0d / state.size();
		this.qLearn(root);
	}

	/**
	 * Collect the winners based on all possible tie-breaking rules
	 * 
	 * @param profile
	 * @return all possible winners
	 */
	public List<Integer> getAllWinners(Profile<Integer> profile) {
		Integer[] items = profile.getSortedItems();
		List<Integer> alternatives = new ArrayList<>(Arrays.asList(items));
		this.profile = preprocess(profile, alternatives);
		this.root = new Node(alternatives);

		this.numNode = 0;
		this.numWinner = 0;

		this.visited = new HashSet<>();
		this.freq = new HashMap<>();
		for (int i = 0; i < numItemTotal; i++)
			freq.put(i, 0);
		this.rewards = new ArrayList<>();

		LinkedList<Node> fringe = new LinkedList<>();
		fringe.add(root);

		trace = new HashMap<>();

		Node next = null;
		while (!fringe.isEmpty()) {
			for (Node node : fringe)
				node.q = getQ(node);

			fringe.sort(comparator);
			next = fringe.pollLast();

			List<Integer> state = next.state;
			if (isLeaf(next) > 0 || hasVisited(state) || canPrune(state)) {
				next.leaf = true;
				continue;
			}

			int[] scores = scoring(profile, state);
			next.children = new ArrayList<>();
			int[] min = MathLib.argmin(scores);
			for (int i : min) {
				Node child = new Node(state, state.get(i));
				child.depth = next.depth + 1;
				next.children.add(child);
			}
			fringe.addAll(next.children);
			visited.add(state);
			numNode++;
		}
		System.out.println(trace);

		List<Integer> winners = new ArrayList<>();
		for (int item : freq.keySet())
			if (freq.get(item) > 0)
				winners.add(item);
		return winners;
	}

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/users/chjiang/github/csc/";
		String dataset = "soc-3";

		NetSTV rule = new NetSTV();
		rule.numItemTotal = 10;

		// List<Path> files = Files.list(Paths.get(base +
		// dataset)).collect(Collectors.toList());
		// Profile<Integer> profile;
		//
		// OpenOption[] options = { StandardOpenOption.APPEND,
		// StandardOpenOption.CREATE, StandardOpenOption.WRITE };
		// StringBuffer sb = null;
		// for (Path file : files) {
		// String name = file.toFile().getName();
		// int m = Integer.parseInt(name.substring(1, name.indexOf("N")));
		// if (m > 10)
		// continue;
		//
		// profile = DataEngine.loadProfile(file);
		// }

		Profile<Integer> profile = DataEngine.loadProfile(Paths.get(base + dataset + "/M10N40-89.csv"));
		rule.qLearn(profile);
		List<Integer> winners = rule.getAllWinners(profile);
		System.out.println(winners);

		TickClock.stopTick();
	}
}
