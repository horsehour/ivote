package com.horsehour.vote.rule.multiround;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 3:06:45 AM, Jan 10, 2017
 *
 */

public class STVPlus {
	public int numItemTotal;
	public Set<List<Integer>> visited;
	public Map<Integer, Integer> freq;
	public List<Integer> electedSampling;

	public Node root;

	public float time, timeSampling, timeScoring;
	public float timeCacheEval, timePruneEval, timeSelectNext, timeHeuristicEval, timeFork;
	public float timePred, timeComputePriority;

	public long begin;

	public Profile<Integer> profile;

	/**
	 * numFailH - number of times that the heuristic fails to work and eliminate
	 * any candidate, numSingleL - number of times to invoke the heuristic and
	 * to eliminate one single loser, numMultiL - number of times to invoke the
	 * heuristic and eliminate multiple losers one time
	 */
	public int numFailH, numSingleL, numMultiL;

	/**
	 * number of nodes produced with the heuristic, which is equal to the sum of
	 * numSingleL and numMultiL; number of nodes produced without the heurisitc
	 */
	public int numNodeWH, numNodeWOH;

	/**
	 * number of nodes expanded, and number of nodes if complete expanded, which
	 * is larger than numNode.
	 */
	public int numNode, numNodeFull;

	/**
	 * number of times to check the hash set to see if the current state has
	 * been visited (hit or miss); number of times to check whether all the
	 * candidates in the current state are winners (hit or miss)
	 */
	public int numCacheHit, numPruneHit;
	public int numCacheMiss, numPruneMiss;

	/**
	 * number of times to compute the plurality score
	 */
	public int numScoring;

	public boolean heuristic, cache, pruning, sampling;
	/**
	 * training model for priority queue
	 */
	public boolean queue;
	/**
	 * recursive or stack (or fringe) version
	 */
	public boolean recursive;

	public String base = "/Users/chjiang/Documents/csc/";
	public String dataset = "soc-3";

	public STVPlus(boolean h, boolean c, boolean p, boolean s, boolean q, boolean r) {
		this.heuristic = h;
		this.cache = c;
		this.pruning = p;
		this.sampling = s;
		this.queue = q;
		this.recursive = r;
	}

	public class Node {
		public List<Integer> state;
		public List<Double> pred;
		public double priority = 0;

		public Node(List<Integer> candidates) {
			this.state = new ArrayList<>(candidates);
		}

		public Node(List<Integer> candidates, Integer eliminated) {
			this.state = new ArrayList<>(candidates);
			this.state.remove(eliminated);
		}

		/**
		 * Update the priority of current node
		 * 
		 * TODO: the number of candidates who are not in the known-winner set
		 * should also be considered
		 * 
		 * @param node
		 */
		void updatePriority() {
			if (pred == null)
				pred = predict(profile, state);

			long start = System.nanoTime();
			priority = 0;
			for (int i = 0; i < state.size(); i++)
				if (freq.get(state.get(i)) == 0)
					priority += pred.get(i);
			priority += 1.0d / state.size();
			timeComputePriority += System.nanoTime() - start;
		}
	}

	/**
	 * Make prediction using the learned algorithm
	 * 
	 * @param profile
	 * @param items
	 * @return prediction score of being selected for all the alternatives
	 */
	List<Double> predict(Profile<Integer> profile, List<Integer> items) {
		if (items.size() == 1)
			return Arrays.asList(5.0);

		long start = System.nanoTime();
		double[][] features = DataEngine.getFeatures(profile, items);
		List<Double> p = new ArrayList<>();
		for (double[] x : features)
			p.add(predict(x));

		// double sum = MathLib.Data.sum(p);
		// p = MathLib.Matrix.multiply(p, 1.0 / sum);

		int[] argmax = MathLib.argmax(p);
		double max = p.get(argmax[0]);
		for (int i = 0; i < items.size(); i++) {
			if (p.get(i) < max)
				p.set(i, 0.0);
		}

		timePred += System.nanoTime() - start;
		return p;
	}

	double[][] w = { { 1.0103063484121213, 0.9696350124299993, 1.0006123390136652 },
			{ 0.9896936516037581, 1.0303649875797136, 0.999387660985303 } };
	double[] b = { 0.012000000569969416, -0.012000000569969416 };

	/**
	 * @param x
	 * @return the probability of being positive
	 */
	double predict(double[] x) {
		double[] predict = new double[2];
		for (int c = 0; c < 2; c++)
			predict[c] = MathLib.Matrix.dotProd(w[c], x) + b[c];
		return predict[1];
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
		numScoring++;
		long start = System.nanoTime();

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
		timeScoring += System.nanoTime() - start;
		return scores;
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
		long start = System.nanoTime();
		for (int item : state) {
			// some candidates is not in the known-winner set
			if (freq.get(item) == 0) {
				timePruneEval += System.nanoTime() - start;
				numPruneMiss++;
				return false;
			}
		}
		numPruneHit++;
		visited.add(state);
		timePruneEval += System.nanoTime() - start;
		return true;
	}

	/**
	 * Evaluate whether the state has been visited
	 * 
	 * @param state
	 * @return true if the state has been visited, false otherwise
	 */
	boolean hasVisited(List<Integer> state) {
		long start = System.nanoTime();
		if (visited.contains(state)) {
			numCacheHit++;
			timeCacheEval += System.nanoTime() - start;
			return true;
		} else {
			numCacheMiss++;
			timeCacheEval += System.nanoTime() - start;
			return false;
		}
	}

	/**
	 * According to the heuristic strategy, we search all possible candidates
	 * that receive least votes and there is a set of candidates, each of them
	 * have the same plurality score and larger than the sum scores of those
	 * candidates that receive few amount of scores. No matter which
	 * tie-breaking rule is used, these candidates or heurisitc loser must be
	 * swept out.
	 * 
	 * @param candidates
	 * @param scores
	 * @param tiers
	 * @return losers according to the heuristic strategy
	 */
	List<Integer> getHeuristicLoser(List<Integer> candidates, int[] scores, TreeMap<Integer, List<Integer>> tiers) {
		long start = System.nanoTime();
		tiers.clear();
		for (int i = 0; i < scores.length; i++) {
			int score = scores[i];
			List<Integer> member = tiers.get(score);
			if (member == null)
				member = new ArrayList<>();
			member.add(candidates.get(i));
			tiers.put(score, member);
		}

		// all candidates have the same score
		if (tiers.size() == 1) {
			timeHeuristicEval += System.nanoTime() - start;
			return null;
		}

		List<Integer> distinct = new ArrayList<>(tiers.keySet());
		int i = 0;
		int remaining = MathLib.Data.sum(scores);
		for (; i < distinct.size(); i++) {
			int score = distinct.get(i);
			int scoreLocal = score * tiers.get(score).size();
			remaining -= scoreLocal;
			if (remaining < score)
				break;
		}

		// no tier satisfies the heuristic elimination criterion
		if (i == distinct.size() - 1) {
			timeHeuristicEval += System.nanoTime() - start;
			return null;
		}

		i += 1;
		List<Integer> losers = new ArrayList<>();
		for (; i < distinct.size(); i++) {
			int score = distinct.get(i);
			losers.addAll(tiers.get(score));
		}
		timeHeuristicEval += System.nanoTime() - start;
		return losers;
	}

	/**
	 * Eliminate candidates directly without considering the relative ordering
	 * to eliminate them
	 * 
	 * @param state
	 * @param scores
	 * @param tiers
	 * @return null if an alternative has been decleared as the winner, the
	 *         elimination is terminated immediately.
	 */
	int[] eliminateHeuristic(List<Integer> state, int[] scores, final TreeMap<Integer, List<Integer>> tiers) {
		List<Integer> losers = null;
		int count = 0;
		while ((losers = getHeuristicLoser(state, scores, tiers)) != null) {
			long start = System.nanoTime();

			numNodeWH++;
			numNodeWOH += losers.size();

			if (losers.size() > 1) {
				numMultiL++;
			} else
				numSingleL++;

			if (cache)
				visited.add(new ArrayList<>(state));

			state.removeAll(losers);

			if (isUniqueRemaining(state) || (cache && hasVisited(state)) || (pruning && canPrune(state)))
				return null;

			if (count > 0)
				numNode++;

			timeHeuristicEval += System.nanoTime() - start;
			scores = scoring(profile, state);
			count++;
		}
		numFailH++;
		return scores;
	}

	/**
	 * Remove candidates who have never been the first choice of any voter and
	 * reconstruct the profile
	 * 
	 * @param profile
	 * @param candidates
	 * @return reconstructed profile if there are any candidate has being
	 *         eliminated, elsewise return the original profile
	 */
	Profile<Integer> preprocess(Profile<Integer> profile, List<Integer> candidates) {
		int[] scores = scoring(profile, candidates);
		List<Integer> eliminated = new ArrayList<>();
		for (int i = 0; i < scores.length; i++) {
			if (scores[i] == 0)
				eliminated.add(i);
		}
		if (eliminated.isEmpty())
			return profile;
		else {
			candidates.removeAll(eliminated);
			return profile.reconstruct(candidates);
		}
	}

	/**
	 * Evalute the current state has only one candidate
	 * 
	 * @param state
	 * @return true if the current state contains only one candidate
	 */
	boolean isUniqueRemaining(List<Integer> state) {
		if (state.size() == 1) {
			int item = state.get(0);
			int count = freq.get(item);
			if (count == 0) {
				numNode++;
				visited.add(state);
			}
			freq.put(item, count + 1);
			return true;
		}
		return false;
	}

	/**
	 * Evalute whether the current state contains an unique winner who gets the
	 * majority votes
	 * 
	 * @param state
	 * @param scores
	 * @return true if the candidate with the largest score, and also gets the
	 *         majority votes, false otherwise
	 */
	boolean containsUniqueWinner(List<Integer> state, int[] scores) {
		int[] max = MathLib.argmax(scores);
		if (scores[max[0]] > profile.numVoteTotal / 2) {
			numNode++;
			int item = state.get(max[0]);
			freq.put(item, freq.get(item) + 1);

			if (cache)
				visited.add(state);
			return true;
		}
		return false;
	}

	void elect() {
		LinkedList<Node> fringe = new LinkedList<>();
		fringe.add(root);

		TreeMap<Integer, List<Integer>> tiers = null;
		Node next = null;
		while (!fringe.isEmpty()) {
			if (queue) {
				for (Node node : fringe)
					node.updatePriority();
				fringe.sort((nd1, nd2) -> Double.compare(nd1.priority, nd2.priority));
			}
			next = fringe.pollLast();

			List<Integer> state = next.state;
			if (isUniqueRemaining(state) || (cache && hasVisited(state)) || (pruning && canPrune(state)))
				continue;

			int[] scores = scoring(profile, state);

			List<Node> children = new ArrayList<>();
			long start = 0;
			if (heuristic) {
				tiers = new TreeMap<>(Collections.reverseOrder());
				List<Integer> losers = getHeuristicLoser(state, scores, tiers);
				if (losers != null) {
					numNodeWH++;
					numNodeWOH += losers.size();

					if (losers.size() > 1)
						numMultiL++;
					else
						numSingleL++;

					start = System.nanoTime();
					visited.add(state);
					timeCacheEval += System.nanoTime() - start;

					start = System.nanoTime();
					List<Integer> childState = new ArrayList<>(state);
					childState.removeAll(losers);
					// add grandchildren
					children.add(new Node(childState));
					timeFork += System.nanoTime() - start;
					numNodeFull++;
				} else {
					// no candidates can be immediately eliminated
					numFailH++;
					// generate children by reusing the created tiers
					start = System.nanoTime();
					for (int item : tiers.lastEntry().getValue())
						children.add(new Node(state, item));
					timeFork += System.nanoTime() - start;
				}
			} else {
				int[] min = MathLib.argmin(scores);
				numNodeFull += min.length;
				// generate children based on their plurality scores
				start = System.nanoTime();
				for (int i : min)
					children.add(new Node(state, state.get(i)));
				timeFork += System.nanoTime() - start;
			}

			// add all children to fringe
			fringe.addAll(children);

			if (cache)
				visited.add(state);
			numNode++;
		}
	}

	/**
	 * Recursion version
	 * 
	 * @param node
	 */
	void elect(Node node) {
		List<Integer> state = node.state;
		if (isUniqueRemaining(state) || (cache && hasVisited(state)) || (pruning && canPrune(state)))
			return;

		int[] scores = scoring(profile, state);
		TreeMap<Integer, List<Integer>> tiers = null;
		if (heuristic) {
			tiers = new TreeMap<>(Collections.reverseOrder());
			scores = eliminateHeuristic(state, scores, tiers);
			if (scores == null)
				return;
		}

		long start = System.nanoTime();
		List<Integer> tied = null;
		if (tiers == null) {
			int[] min = MathLib.argmin(scores);
			tied = new ArrayList<>();
			for (int i : min)
				tied.add(state.get(i));
		} else {
			int smallest = tiers.lastKey();
			tied = tiers.get(smallest);
		}
		timeSelectNext += System.nanoTime() - start;

		int numElim = tied.size();
		numNodeFull += numElim;

		Node child;
		while ((numElim = tied.size()) > 0) {
			start = System.nanoTime();
			int highest = -1, index = -1;
			// priority selection
			for (int i = 0; i < numElim; i++) {
				int f = freq.get(tied.get(i));
				if (f > highest) {
					index = i;
					highest = f;
				}
			}
			int next = tied.remove(index);
			timeSelectNext += System.nanoTime() - start;

			start = System.nanoTime();
			child = new Node(state, next);
			timeFork += System.nanoTime() - start;

			elect(child);

			if (cache && !hasVisited(child.state)) {
				start = System.nanoTime();
				numNode++;
				visited.add(child.state);
				timeCacheEval += System.nanoTime() - start;
			}

			if (pruning && canPrune(state))
				break;
		}

		if (cache)
			visited.add(state);
	}

	/**
	 * @param profile
	 * @return All possible winners with various tie-breaking rules
	 */
	public List<Integer> getAllWinners(Profile<Integer> profile) {
		this.visited = new HashSet<>();

		this.numNode = 0;
		this.numNodeWH = 0;
		this.numNodeWOH = 0;
		this.numNodeFull = 0;

		this.numFailH = 0;
		this.numSingleL = 0;
		this.numMultiL = 0;

		this.numScoring = 0;
		this.numCacheHit = 0;
		this.numPruneHit = 0;
		this.numCacheMiss = 0;
		this.numPruneMiss = 0;

		Integer[] items = profile.getSortedItems();

		this.timeScoring = 0;
		this.timeCacheEval = 0;
		this.timePruneEval = 0;
		this.timeSelectNext = 0;
		this.timeHeuristicEval = 0;
		this.timeFork = 0;
		this.timePred = 0;
		this.timeComputePriority = 0;

		List<Integer> state = new ArrayList<>(Arrays.asList(items));
		this.profile = preprocess(profile, state);

		freq = new HashMap<>();
		BruteForceSTV stvBF = null;
		if (sampling) {
			stvBF = new BruteForceSTV();
			stvBF.getAllWinners(profile, state.size());
			for(Integer item : state)
				freq.put(item, stvBF.countElected.get(item));
		} else {
			for (Integer item : state)
				freq.put(item, 0);
		}

		this.begin = System.nanoTime();
		this.numItemTotal = state.size();
		this.root = new Node(state);

		if (recursive)
			elect(root);
		else
			elect();
		time = (System.nanoTime() - begin) / 1e9f;

		timeScoring /= 1e9f;
		timeCacheEval /= 1e9f;
		timePruneEval /= 1e9f;
		timeSelectNext /= 1e9f;
		timeHeuristicEval /= 1e9f;
		timeFork /= 1e9f;
		timePred /= 1e9f;
		timeComputePriority /= 1e9f;

		if (sampling) {
			timeSampling = stvBF.elapsed;
			electedSampling = stvBF.elected;
		}
		List<Integer> winners = new ArrayList<>();
		for (int item : freq.keySet())
			if (freq.get(item) > 0)
				winners.add(item);
		return winners;
	}

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/Users/chjiang/Documents/csc/";
		String dataset = "soc-3";

		boolean heuristic = false, cache = true, pruning = true;
		boolean sampling = false, queue = true, recursive = false;

		STVPlus rule = new STVPlus(heuristic, cache, pruning, sampling, queue, recursive);
		Path input = Paths.get(base + dataset + "/M20N20-38.csv");

		Profile<Integer> profile = DataEngine.loadProfile(input);
		List<Integer> winners = rule.getAllWinners(profile);

		String format = "#node=%d, #score=%d, #cache=%d, #t=%f, t_score=%f, winners=%s\n";
		System.out.printf(format, rule.numNode, rule.numScoring, rule.visited.size(), rule.time, rule.timeScoring,
				winners);

		TickClock.stopTick();
	}
}
