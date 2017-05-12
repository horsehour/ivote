package com.horsehour.vote.rule.multiround;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;
import com.horsehour.vote.data.FeatureLab;

/**
 * STV learned with Reinforcement learning
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 3:06:45 AM, Jan 10, 2017
 */

public class RLSTV extends PerfectSTV {
	public int numItemTotal;
	public int numWinner, numNode;

	public int nSample = 0, maxSample = 50;

	public Map<Integer, Integer> freq;

	public double alpha = 0.1, gamma = 0.8;
	public double unit = 0;

	public double[] qFunction;
	public double[] features;

	public Node optimal;

	public RLSTV() {}

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
				visited.put(node.state, node);
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
	boolean canPrune(Node node) {
		for (int item : node.state) {
			// some candidates is not in the known-winner set
			if (freq.get(item) == 0) {
				return false;
			}
		}
		visited.put(node.state, node);
		return true;
	}

	/**
	 * Evaluate whether the node has been visited
	 * 
	 * @param node
	 * @return
	 */
	boolean hasVisited(Node node) {
		Node prev = visited.get(node.state);
		if (prev != null)
			return true;
		else
			return false;
	}

	/**
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
			boolean flag = hasVisited(node);
			if (flag || canPrune(node))
				reward -= 5 * unit;
			else if (!flag) {
				visited.put(node.state, node);
			}
		}
		return reward;
	}

	/**
	 * @return reward received from an episode
	 */
	public double getReward() {
		double nom = 0, denom = 0;
		for (int i = 1; i <= trace.size(); i++) {
			nom += trace.get(1) * 1.0d / trace.get(i);
			denom += 1.0d / i;
		}
		return nom / denom;
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

	/***
	 * Update the Q-function
	 * 
	 * @param node
	 */
	public void updateQ(Node node, Node next) {
		double diff;
		if (next == null)
			diff = getReward() + gamma * 0 - node.q;
		else
			diff = getReward(node) + gamma * next.q - node.q;

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
				nSample++;
				// System.out.println(++nSample + " : " + getReward(node));
				if (nSample < maxSample) {} else
					break;
			}
		}
	}

	/**
	 * Sample-based value iteration
	 * 
	 * @param episode
	 */
	public void qLearn(List<Node> episode) {
		Node node, next;
		for (int i = 0; i < episode.size(); i++) {
			node = episode.get(i);
			node.q = getQ(node);
		}

		for (int i = 0; i < episode.size(); i++) {
			node = episode.get(i);
			if (i == episode.size() - 1)
				next = null;
			else
				next = episode.get(i + 1);
			updateQ(node, next);
		}
	}

	/**
	 * Q-Learning
	 * 
	 * @param profile
	 */
	public void qLearn(Profile<Integer> profile) {
		this.visited = new HashMap<>();
		this.freq = new HashMap<>();
		for (int i = 0; i < numItemTotal; i++)
			freq.put(i, 0);

		this.trace = new HashMap<>();

		Integer[] items = profile.getSortedItems();
		List<Integer> state = new ArrayList<>(Arrays.asList(items));
		this.profile = preprocess(profile, state);
		this.root = new Node(state);

		features = FeatureLab.getF1(profile, state, numItemTotal);
		if (qFunction == null)
			qFunction = new double[features.length];
		MathLib.Rand.distribution(qFunction);
		this.unit = numItemTotal * 1.0d / state.size();
		this.qLearn(root);
	}

	/**
	 * Q-value function approximation
	 * 
	 * @param profile
	 * @param nSample
	 */
	public void qLearn(Profile<Integer> profile, int nSample) {
		this.visited = new HashMap<>();
		this.freq = new HashMap<>();
		for (int i = 0; i < numItemTotal; i++)
			freq.put(i, 0);

		this.trace = new HashMap<>();

		Integer[] items = profile.getSortedItems();
		List<Integer> cands = new ArrayList<>(Arrays.asList(items));
		this.profile = preprocess(profile, cands);
		this.root = new Node(cands);
		this.leaves = new ArrayList<>();

		features = FeatureLab.getF1(profile, cands, numItemTotal);
		if (qFunction == null)
			qFunction = new double[features.length];

		MathLib.Rand.distribution(qFunction);
		this.unit = numItemTotal * 1.0d / cands.size();

		/** build up a search tree **/
		this.numNode = 0;
		this.numWinner = 0;

		this.dfs();

		/** train with sample **/
		List<List<Node>> episodes = getEpisodes(nSample);
		for (List<Node> episode : episodes)
			qLearn(episode);
	}

	/***
	 * Learn the value function
	 * 
	 * @param files
	 * @param nSample
	 * @return value function
	 * @throws IOException
	 */
	public double[] train(List<Path> files, int nSample) throws IOException {
		Profile<Integer> profile = null;
		for (Path file : files) {
			profile = DataEngine.loadProfile(file);
			qLearn(profile, nSample);
		}
		return qFunction;
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

		this.visited = new HashMap<>();
		this.freq = new HashMap<>();
		for (int i = 0; i < numItemTotal; i++)
			freq.put(i, 0);

		LinkedList<Node> fringe = new LinkedList<>();
		fringe.add(root);

		trace = new HashMap<>();

		Node next = null;
		while (!fringe.isEmpty()) {
			for (Node node : fringe)
				node.q = getQ(node);

			fringe.sort(comparator);
			next = fringe.pollLast();

			if (isLeaf(next) > 0 || hasVisited(next) || canPrune(next)) {
				next.leaf = true;
				continue;
			}

			List<Integer> state = next.state;
			int[] scores = scoring(profile, state);
			next.children = new ArrayList<>();
			int[] min = MathLib.argmin(scores);
			for (int i : min) {
				Node child = new Node(state, state.get(i));
				child.depth = next.depth + 1;
				next.children.add(child);
			}
			fringe.addAll(next.children);
			visited.put(state, next);
			numNode++;
		}
		System.out.println(trace);

		List<Integer> winners = new ArrayList<>();
		for (int item : freq.keySet())
			if (freq.get(item) > 0)
				winners.add(item);
		return winners;
	}

	public static void main1(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/users/chjiang/github/csc/";
		String dataset = "soc-6-stv";

		RLSTV rule = new RLSTV();
		rule.numItemTotal = 10;

		int nSample = 100;

		List<Path> inputs = new ArrayList<>();
		Path source = Paths.get(base + dataset);
		Files.list(source).forEach(input -> {
			String name = input.toFile().getName();
			int ind = name.indexOf("N");
			int m = Integer.parseInt(name.substring(1, ind));
			if (m > 10)
				return;
			inputs.add(input);
		});

		double[] model = rule.train(inputs, nSample);
		System.out.println(Arrays.toString(model));

		Profile<Integer> profile = DataEngine.loadProfile(Paths.get(base + dataset + "/M10N40-89.csv"));
		rule.getAllWinners(profile);

		TickClock.stopTick();
	}
}