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
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;

import org.apache.commons.lang3.SerializationUtils;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;
import com.horsehour.vote.data.FeatureLab;
import com.horsehour.vote.data.VoteExp;

import smile.classification.SoftClassifier;

/**
 * STV with Boost model in assistance to make prediction
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 3:06:45 AM, Jan 10, 2017
 */

public class BoostSTV {
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
	public SoftClassifier<double[]> algo;

	public Map<Integer, Integer> trace;
	public int numWinner;
	public int rank = 0;

	public int numNode;

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

	public boolean cache, pruning, sampling;
	/**
	 * priority function {0, 1, 2, ...}, and pf = 0 indicates priority function
	 * is not used or turn-off
	 */
	public int pFunction;

	public String base = "/users/chjiang/github/csc/";
	public String dataset = "soc-3";

	public BoostSTV() {}

	public BoostSTV(boolean c, boolean p, boolean s, int pf) {
		this.cache = c;
		this.pruning = p;
		this.sampling = s;
		this.pFunction = pf;
	}

	public class Node {
		public List<Integer> state;
		public List<Double> pred;

		public int[] scoresBorda;
		public int[] scoresPlurality;

		public double priority = 0;

		public boolean leaf = false;
		public int depth = 0;

		/** visited order **/
		public int order;

		public List<Node> children;
		public Node parent;

		public List<Integer> winners;
		public List<Integer> winnersPredicted;

		public Node(List<Integer> candidates) {
			this.state = new ArrayList<>(candidates);
		}

		public Node(List<Integer> candidates, Integer eliminated) {
			this.state = new ArrayList<>(candidates);
			this.state.remove(eliminated);
		}
	}

	public VoteExp exp;

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
				if (freq.get(node.state.get(i)) == 0)
					expectation += node.pred.get(i);
			node.priority *= expectation;
		} else if (pFunction == 2) {
			node.priority = 0;
			for (int i = 0; i < m; i++)
				if (freq.get(node.state.get(i)) == 0)
					node.priority += 1;
			node.priority += (numItemTotal - m) * numItemTotal;
		} else if (pFunction == 4) {
			node.priority = 0;
			if (node.scoresPlurality == null)
				node.scoresPlurality = scoring(profile, node.state);
			for (int i = 0; i < m; i++)
				if (freq.get(node.state.get(i)) == 0)
					node.priority += node.scoresPlurality[i];
			node.priority /= profile.numVoteTotal;
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
				node.pred = predict(profile, node.state);

			node.priority = (numItemTotal - m) * numItemTotal;
			double expectation = 0;
			for (Integer i = 0; i < numItemTotal; i++) {
				int ind = node.state.indexOf(i);
				if (ind > -1 && freq.get(i) == 0)
					expectation += node.pred.get(i);
			}
			node.priority *= expectation;
		} else if (pFunction == 10) {
			if (node.pred == null)
				node.pred = predict(profile, node.state);
			double expectation = 0;
			for (Integer i = 0; i < numItemTotal; i++) {
				int ind = node.state.indexOf(i);
				if (ind > -1 && freq.get(i) == 0)
					expectation += node.pred.get(i);
			}
			node.priority = expectation;
		} else if (pFunction == 11) {
			// compare between children's prediction and parent's winner,
			// if the prediction is consistent with new winners, the new winners
			// indicates a high priority. if there is no new winner, we should
			// refer to other states for freshness.

			// TODO: Apt 2, 2017
			node.priority = 0;
			// parent does not have any winner
			if (node.parent.winners == null) {
				// based on the prediction from parent
				for (int w : node.winnersPredicted) {
					if (!node.parent.winnersPredicted.contains(w)) {
						node.priority++;
					}
				}
			} else {
				for (int w : node.winnersPredicted) {
					if (!node.parent.winners.contains(w))
						node.priority++;
				}
			}
			// continue the path and explore
			if (node.priority > 0)
				update = false;
		} else if (pFunction == 12) {
			// TODO: Apt 2, 2017
			node.priority = 0;
			for (int w : node.winnersPredicted) {
				if (freq.get(w) == 0)
					node.priority++;
			}

			// continue the path and explore
			if (node.priority > 0) {
				node.priority /= node.state.size();
				update = false;
			}
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
	 * Make prediction using the learned algorithm
	 * 
	 * @param profile
	 * @param items
	 * @return prediction scores over all alternatives
	 */
	List<Double> predict(Profile<Integer> profile, List<Integer> items) {
		List<Double> p = new ArrayList<>(Collections.nCopies(numItemTotal, 0.0));
		if (items.size() == 1) {
			p.set(items.get(0), 1.0);
			return p;
		}

		long start = System.nanoTime();
		double[] posteriori = new double[numItemTotal];
		double[] features = FeatureLab.getF1(profile, items, numItemTotal);
		algo.predict(features, posteriori);

		// adjust the predictions such that those not included are assigned zero
		for (int i = 0; i < numItemTotal; i++) {
			int ind = items.indexOf(i);
			if (ind == -1)
				posteriori[i] = 0;
		}

		MathLib.Scale.sum(posteriori);

		// TODO: biggest bug!!!!!
		// for (int i = 0; i < posteriori.length; i++)
		// p.add(posteriori[i]);

		for (int i = 0; i < posteriori.length; i++)
			p.set(i, posteriori[i]);

		timePred += System.nanoTime() - start;
		return p;
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
	 * Remove candidates who have never been the first choice of any voter and
	 * reconstruct the profile
	 * 
	 * @param profile
	 * @param state
	 * @return reconstructed profile if there are any candidate has being
	 *         eliminated, elsewise return the original profile
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
	 * @since Apt 2, 2017
	 * @param node
	 * @return
	 */
	int isUniqueRemaining(Node node) {
		List<Integer> state = node.state;
		int signal = 0;
		if (state.size() == 1) {
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
	 * TODO Propagate winner that recently found to the parent nodes up to the
	 * root
	 * 
	 * @param node
	 * @param winner
	 * @since Apt 2, 2017
	 */
	void upwardPropagate(Node node, int winner) {
		if (node != null) {
			if (node.winners == null) {
				node.winners = new ArrayList<>(Arrays.asList(winner));
			} else
				node.winners.add(winner);
			upwardPropagate(node.parent, winner);
		}
	}

	Comparator<Node> comparator = (nd1, nd2) -> Double.compare(nd1.priority, nd2.priority);

	// TODO APT 2, 2017
	// count the frequency that certain state being elected, and the one with
	// least number of being predicted should given high priority
	Comparator<Node> comparator2 = (nd1, nd2) -> {
		int v = Double.compare(nd1.priority, nd2.priority);
		if (v == 0)
			v = Integer.compare(nd1.depth, nd2.depth);
		return v;
	};

	public boolean update = false;

	void elect() {
		LinkedList<Node> fringe = new LinkedList<>();
		fringe.add(root);
		double[] features = FeatureLab.getF1(profile, root.state, numItemTotal);
		root.winnersPredicted = exp.mechanism.apply(features, root.state);

		root.order = rank++;
		Node next = null;
		while (!fringe.isEmpty()) {
			if (pFunction > 0 && update) {
				for (Node node : fringe) {
					updatePriority(node);
				}
				fringe.sort(comparator);
				if (pFunction == 12)
					fringe.sort(comparator2);
			}
			next = fringe.pollLast();
			next.order = rank++;

			// TODO since Apt 2, 2017
			List<Integer> state = next.state;
			int signal = isUniqueRemaining(next);
			if (signal > 0 || (cache && hasVisited(state)) || (pruning && canPrune(state))) {
				if (signal > 0) {
					if (pFunction == 12)
						upwardPropagate(next, state.get(0));
				}

				next.leaf = true;
				update = true;

				if (signal == 1)
					continue;

				next.order *= -1;
				rank -= 1;
				continue;
			}

			int[] scores = null;
			if (next.scoresPlurality == null)
				scores = scoring(profile, state);
			else
				scores = next.scoresPlurality;

			// List<Node> children = new ArrayList<>();
			next.children = new ArrayList<>();
			long start = 0;
			int[] min = MathLib.argmin(scores);
			start = System.nanoTime();
			for (int i : min) {
				Node child = new Node(state, state.get(i));
				features = FeatureLab.getF1(profile, child.state, numItemTotal);
				child.winnersPredicted = exp.mechanism.apply(features, child.state);

				child.depth = next.depth + 1;
				child.parent = next;
				next.children.add(child);
				// children.add(new Node(state, state.get(i)));
			}

			// TODO Apr 2, 2017
			// when there is no winner, fringe is empty, select the state with
			// most predicted
			// winners.
			if (pFunction == 12 && min.length > 1 && fringe.isEmpty() && numWinner == 0)
				update = true;

			timeFork += System.nanoTime() - start;

			// add all children to fringe
			// fringe.addAll(children);
			fringe.addAll(next.children);

			if (cache)
				visited.add(state);
			numNode++;
		}
		prefMatrix = null;
	}

	// TODO: shuffle based on the predicted winners as used in PerfectSTV
	void shuffle(Node next) {
		int numChild = next.children.size();
		if (numChild == 1)
			return;

		Node child;
		for (int i = 0; i < numChild; i++) {
			child = next.children.get(i);
			if (child.pred == null)
				child.pred = predict(profile, child.state);

			child.priority = 0;

			for (int k = 0; k < numItemTotal; k++) {
				if (freq.get(k) == 0)
					child.priority += child.pred.get(k);
			}
		}

		/** sort based on priority **/
		next.children.sort((nd1, nd2) -> Double.compare(nd1.priority, nd2.priority));
	}

	void elect2() {
		LinkedList<Node> fringe = new LinkedList<>();
		fringe.add(root);
		root.pred = predict(profile, root.state);
		root.order = rank++;

		Node next = null;
		while (!fringe.isEmpty()) {
			next = fringe.pollLast();
			next.order = rank++;

			List<Integer> state = next.state;
			int signal = isUniqueRemaining(next);
			if (signal > 0 || (cache && hasVisited(state)) || (pruning && canPrune(state))) {
				next.leaf = true;
				if (signal == 1)
					continue;

				next.order *= -1;
				rank -= 1;
				continue;
			}

			int[] scores = null;
			if (next.scoresPlurality == null)
				scores = scoring(profile, state);
			else
				scores = next.scoresPlurality;

			next.children = new ArrayList<>();
			long start = 0;
			int[] min = MathLib.argmin(scores);
			start = System.nanoTime();
			for (int i : min) {
				Node child = new Node(state, state.get(i));
				child.pred = predict(profile, child.state);
				child.depth = next.depth + 1;
				child.parent = next;
				next.children.add(child);
			}

			timeFork += System.nanoTime() - start;

			shuffle(next);

			fringe.addAll(next.children);

			if (cache)
				visited.add(state);
			numNode++;
		}
		prefMatrix = null;
	}

	/**
	 * @param profile
	 * @return All possible winners with various tie-breaking rules
	 */
	public List<Integer> getAllWinners(Profile<Integer> profile) {
		this.visited = new HashSet<>();

		this.numNode = 0;
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
		this.profile = profile;

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
			// for (Integer item : state)
			// freq.put(item, 0);
			for (int i = 0; i < numItemTotal; i++)
				freq.put(i, 0);
		}

		this.begin = System.nanoTime();
		// this.numItemTotal = state.size();
		this.root = new Node(state);

		this.trace = new HashMap<>();

		this.elect();
		// this.elect2();

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

	public static void report() throws IOException {
		String base = "/Users/chjiang/github/csc/";
		String dataset = "soc-3-hardcase";

		boolean cache = true, pruning = true, sampling = false;
		int pFunction = 1;

		BoostSTV rule = new BoostSTV(cache, pruning, sampling, pFunction);
		rule.numItemTotal = 30;
		rule.exp = new VoteExp();
		rule.exp.m = rule.numItemTotal;
		rule.exp.cutoff = 0.51;

		Path model = Paths.get(base + "logistic.m.30.mdl");
		rule.algo = SerializationUtils.deserialize(Files.readAllBytes(model));
		rule.exp.algo = rule.algo;

		List<Path> files = Files.list(Paths.get(base + dataset)).collect(Collectors.toList());
		Profile<Integer> profile;

		OpenOption[] options = { StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE };
		StringBuffer sb = null;
		for (Path file : files) {
			String name = file.toFile().getName();
			int m = Integer.parseInt(name.substring(1, name.indexOf("N")));
			if (m != 10)
				continue;

			profile = DataEngine.loadProfile(file);

			rule.pFunction = 0;
			List<Integer> winners = rule.getAllWinners(profile);
			if (winners.size() == 1)
				continue;

			System.out.println(name);
			sb = new StringBuffer().append(name).append("\t");
			List<Integer> v1 = new ArrayList<>(rule.trace.values());
			v1.add(rule.numNode);
			v1.add(rule.numScoring);

			sb.append(v1.toString().replaceAll("\\[|\\]| ", "")).append("\t");

			// sb.append(rule.numNode + "\t" + rule.numScoring + "\t" +
			// winners).append("\t");

			rule.pFunction = 8;
			winners = rule.getAllWinners(profile);

			// sb.append(rule.numNode + "\t" + rule.numScoring).append("\n");
			List<Integer> v2 = new ArrayList<>(rule.trace.values());
			v2.add(rule.numNode);
			v2.add(rule.numScoring);
			sb.append(v2.toString().replaceAll("\\[|\\]| ", "")).append("\t");
			List<Float> perf = new ArrayList<>();
			float imprv = 0;
			for (int i = 1; i < v1.size(); i++) {
				imprv = 1 - v2.get(i) * 1.0F / v1.get(i);
				perf.add(imprv);
			}

			int sz = v1.size();
			sb.append(perf.toString().replaceAll("\\[|\\]| ", "")).append("\t");
			sb.append(v1.get(sz - 1)).append("\t");
			sb.append(v2.get(sz - 1)).append("\n");
			Files.write(Paths.get(base + "diff.csv"), sb.toString().getBytes(), options);
		}
	}

	public static void searchTree() throws IOException {
		String base = "/users/chjiang/github/csc/";
		 String dataset = "soc-3";
//		String dataset = "soc-6-stv";

		boolean cache = true, pruning = true, sampling = false;
		int pFunction = 1;

		BoostSTV rule = new BoostSTV(cache, pruning, sampling, pFunction);
		rule.numItemTotal = 30;
		rule.exp = new VoteExp();
		rule.exp.m = rule.numItemTotal;
		rule.exp.cutoff = 0.51;

		Path model = Paths.get(base + "logistic.m.30.mdl");
		rule.algo = SerializationUtils.deserialize(Files.readAllBytes(model));
		rule.exp.algo = rule.algo;

		Path input = Paths.get(base + dataset + "/M10N10-36.csv");
		Profile<Integer> profile = DataEngine.loadProfile(input);
		List<Integer> winners = rule.getAllWinners(profile);

		String format = "#node=%d, #score=%d, #cache=%d, t=%f, t_score=%f, winners=%s\n";
		System.out.printf(format, rule.numNode, rule.numScoring, rule.visited.size(), rule.time, rule.timeScoring,
				winners);
		System.out.println(rule.trace);

		String tree = SearchTree.create(rule.root);
		Files.write(Paths.get(base + "/votetree/stv.json"), tree.getBytes());
	}

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		 searchTree();
//		report();

		TickClock.stopTick();
	}
}