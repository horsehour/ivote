package com.horsehour.vote.rule.multiseat;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.DataEngine;
import com.horsehour.vote.Profile;

import smile.classification.LogisticRegression;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 3:06:45 AM, Jan 10, 2017
 */

public class STVPlus2 {
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
	public LogisticRegression model;

	public Map<Integer, Integer> trace;
	public int numWinner;

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
	 * priority function {0, 1, 2, ...}, and pf = 0 indicates priority function
	 * is not used or turn-off
	 */
	public int pFunction;
	/**
	 * recursive or stack (or fringe) version
	 */
	public boolean recursive;

	public String base = "/Users/chjiang/Documents/csc/";
	public String dataset = "soc-3";

	public STVPlus2() {}

	public STVPlus2(boolean h, boolean c, boolean p, boolean s, boolean r, int pf) {
		this.heuristic = h;
		this.cache = c;
		this.pruning = p;
		this.sampling = s;
		this.pFunction = pf;
		this.recursive = r;
	}

	public class Node {
		public List<Integer> state;
		public List<Double> pred;

		public int[] scoresBorda;
		public int[] scoresPlurality;

		public double priority = 0;

		public Node(List<Integer> candidates) {
			this.state = new ArrayList<>(candidates);
		}

		public Node(List<Integer> candidates, Integer eliminated) {
			this.state = new ArrayList<>(candidates);
			this.state.remove(eliminated);
		}
	}

	/**
	 * Update the priority of current node
	 */
	void updatePriority(Node node) {
		long start = System.nanoTime();
		int m = node.state.size();
		if (pFunction == 1) {
			if (node.pred == null)
				node.pred = predict(profile, node.state);
			node.priority = numItemTotal - m;
			double expectation = 0;
			for (int i = 0; i < m; i++)
				if (freq.get(node.state.get(i)) == 0 && node.pred.get(i) > 0.5)
					expectation += node.pred.get(i);
			node.priority *= expectation;
		} else if (pFunction == 2) {
			node.priority = 0;
			for (int i = 0; i < m; i++)
				if (freq.get(node.state.get(i)) == 0)
					node.priority += 1;
			node.priority += (numItemTotal - m) * numItemTotal;
		} else if (pFunction == 3) {
			node.priority = 0;
			if (node.scoresBorda == null)
				node.scoresBorda = borda(profile, node.state);
			for (int i = 0; i < m; i++)
				if (freq.get(node.state.get(i)) == 0)
					node.priority += node.scoresBorda[i];
			int n = profile.numVoteTotal;
			double maxBorda = m * (m - 1) * n / 2;
			node.priority /= maxBorda;
		} else if (pFunction == 4) {
			node.priority = 0;
			if (node.scoresPlurality == null)
				node.scoresPlurality = scoring(profile, node.state);
			for (int i = 0; i < m; i++)
				if (freq.get(node.state.get(i)) == 0)
					node.priority += node.scoresPlurality[i];
			node.priority /= profile.numVoteTotal;
		} else if (pFunction == 5) {
			if (node.scoresBorda == null)
				node.scoresBorda = borda(profile, node.state);

			if (node.scoresPlurality == null)
				node.scoresPlurality = scoring(profile, node.state);
			double r1 = 0, r2 = 0;
			for (int i = 0; i < m; i++)
				if (freq.get(node.state.get(i)) == 0) {
					r1 += node.scoresPlurality[i];
					r2 += node.scoresBorda[i];
				}
			r1 /= profile.numVoteTotal;
			int n = profile.numVoteTotal;
			double maxBorda = m * (m - 1) * n / 2;
			r2 /= maxBorda;
			node.priority = r1 + r2;
		} else if (pFunction == 6) {
			// leaf has depth 1.0, root has depth 0.0
			double depth = (numItemTotal - m) * 1.0d / numItemTotal;
			double freshness = 0;
			for (int i = 0; i < m; i++)
				if (freq.get(node.state.get(i)) == 0)
					freshness += 1;
			freshness /= m;
			// alpha is the weight of depth, beta is the weight of freshness,
			// and gamma indicates the discount factor of the previous priority
			// estimation, which is still valuable though less valuable than the
			// newest statistics.
			double alpha = numItemTotal * numItemTotal, beta = m, gamma = 0.2;
			node.priority = gamma * node.priority + alpha * depth + beta * freshness;
		} else if (pFunction == 7) {
			// leaf has depth 1.0, root has depth 0.0
			double depth = (numItemTotal - m) * 1.0d / numItemTotal;
			double freshness = 0;
			for (int i = 0; i < m; i++)
				if (freq.get(node.state.get(i)) == 0)
					freshness += 1;
			freshness /= m;
			// alpha is the weight of depth, beta is the weight of freshness,
			// and gamma indicates the discount factor of the previous priority
			// estimation, which is still valuable though less valuable than the
			// newest statistics.
			double alpha = 20, beta = m;
			node.priority = alpha * depth + beta * freshness;
		} else if (pFunction == 8) {
			if (node.pred == null)
				node.pred = predict(profile, node.state);
			node.priority = (numItemTotal - m) * numItemTotal;
			double expectation = 0;
			for (int i = 0; i < m; i++)
				if (freq.get(node.state.get(i)) == 0 && node.pred.get(i) > 0.5)
					expectation += node.pred.get(i);
			node.priority *= expectation;
		} else if (pFunction == 9) {
			if (node.pred == null)
				node.pred = predict2(profile, node.state);
			node.priority = (numItemTotal - m) * numItemTotal;
			double expectation = 0;
			for (int i = 0; i < m; i++)
				if (freq.get(node.state.get(i)) == 0 && node.pred.get(i) > 0.5)
					expectation += node.pred.get(i);
			node.priority *= expectation;
		}
		timeComputePriority += System.nanoTime() - start;
	}
	public Map<Integer, Map<Integer, Integer>> prefMatrix;

	void getPrefMatrix(Profile<Integer> profile) {
		prefMatrix = new HashMap<>();
		int[] votes = profile.votes;
		int m = profile.getNumItem();
		for (int k = 0; k < votes.length; k++) {
			Integer[] pref = profile.data[k];
			for (int i = 0; i < m; i++) {
				Map<Integer, Integer> outlink = prefMatrix.get(pref[i]);
				if (outlink == null) {
					outlink = new HashMap<>();
					prefMatrix.put(pref[i], outlink);
				}
				for (int j = i + 1; j < m; j++) {
					if (outlink.get(pref[j]) == null)
						outlink.put(pref[j], votes[k]);
					else
						outlink.put(pref[j], outlink.get(pref[j]) + votes[k]);
				}
			}
		}
	}

	/**
	 * Compute alternatives' Borda scores based on preference matrix
	 * 
	 * @param profile
	 * @param state
	 * @return Borda scores
	 */
	int[] borda(Profile<Integer> profile, List<Integer> state) {
		if (prefMatrix == null)
			getPrefMatrix(profile);

		int m = state.size();
		int[] scores = new int[m];

		Map<Integer, Integer> outlink;
		for (int i = 0; i < m; i++) {
			int item1 = state.get(i);
			outlink = prefMatrix.get(item1);
			for (int item2 : state) {
				if (item2 == item1 || outlink.get(item2) == null)
					continue;
				scores[i] += outlink.get(item2);
			}
		}
		return scores;
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
		timePred += System.nanoTime() - start;
		return p;
	}

	List<Double> predict2(Profile<Integer> profile, List<Integer> items) {
		if (items.size() == 1)
			return Arrays.asList(5.0);

		long start = System.nanoTime();
		double[][] features = DataEngine.getFeatures(profile, items);
		List<Double> p = new ArrayList<>();
		for (double[] x : features) {
			double[] posteriori = new double[2];
			model.predict(x, posteriori);
			p.add(posteriori[1]);
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
		return 1.0d / (1 + Math.exp(predict[0] - predict[1]));
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
	 * @param state
	 * @return reconstructed profile if there are any candidate has being
	 *         eliminated, elsewise return the original profile
	 */
	Profile<Integer> preprocess(Profile<Integer> profile, List<Integer> state) {
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
				numWinner++;
				visited.add(state);
				trace.put(numWinner, numNode);
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

	Comparator<Node> comparator = (nd1, nd2) -> Double.compare(nd1.priority, nd2.priority);

	void elect() {
		LinkedList<Node> fringe = new LinkedList<>();
		// PriorityQueue<Node> fringe = new PriorityQueue<Node>(10, comparator);
		fringe.add(root);
		trace.put(numNode, numWinner);
		TreeMap<Integer, List<Integer>> tiers = null;
		Node next = null;
		while (!fringe.isEmpty()) {
			if (pFunction > 0) {
				for (Node node : fringe)
					updatePriority(node);
				fringe.sort(comparator);
			}
			next = fringe.pollLast();
			// next = fringe.poll();

			List<Integer> state = next.state;
			if (isUniqueRemaining(state) || (cache && hasVisited(state)) || (pruning && canPrune(state)))
				continue;

			int[] scores = null;
			if (next.scoresPlurality == null)
				scores = scoring(profile, state);
			else
				scores = next.scoresPlurality;

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
		prefMatrix = null;
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
		BruteForceSTV ruleBF = null;
		if (sampling) {
			ruleBF = new BruteForceSTV();
			int m = state.size();
			List<Integer> winners = ruleBF.getAllWinners(profile, m);
			for (Integer item : state)
				freq.put(item, ruleBF.countElected.get(item));
			this.numWinner = winners.size();
		} else {
			this.numWinner = 0;
			for (Integer item : state)
				freq.put(item, 0);
		}

		this.begin = System.nanoTime();
		this.numItemTotal = state.size();
		this.root = new Node(state);

		this.trace = new HashMap<>();
		this.trace.put(numWinner, numNode);
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
			timeSampling = ruleBF.elapsed;
			electedSampling = ruleBF.elected;
		}

		List<Integer> winners = new ArrayList<>();
		for (int item : freq.keySet())
			if (freq.get(item) > 0)
				winners.add(item);
		return winners;
	}

	public static void main13(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/Users/chjiang/Documents/csc/";
		String dataset = "soc-5-stv";

		boolean heuristic = false, cache = true, pruning = true;
		boolean sampling = false, recursive = false;
		int pFunction = 0;

		STVPlus2 rule = new STVPlus2(heuristic, cache, pruning, sampling, recursive, pFunction);
		Path input = Paths.get(base + dataset + "/M80N70-23.csv");
		Profile<Integer> profile = DataEngine.loadProfile(input);
		List<Integer> winners = rule.getAllWinners(profile);

		String format = "#node=%d, #score=%d, #cache=%d, #success=%d, t=%f, t_score=%f, winners=%s\n";
		System.out.printf(format, rule.numNode, rule.numScoring, rule.visited.size(), rule.numNodeWH - rule.numSingleL,
				rule.time, rule.timeScoring, winners);

		TickClock.stopTick();
	}
}
