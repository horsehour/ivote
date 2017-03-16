package com.horsehour.vote.rule.multiseat;

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
import com.horsehour.vote.DataEngine;
import com.horsehour.vote.Profile;

/**
 * Single transferable voting method (STV, a.k.a Cincinnati rule or Hare voting)
 * eliminates no more than one candidate at a time. STV with single winner is
 * sometimes called "alternative voting". When there are multiple winners, the
 * algorithm for STV becomes considerably complex. The implementation tries to
 * have all the winners based on each possible tie-breaking rule.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 1:10:41 PM, Dec 21, 2016
 */
public class STV {
	public int numItemTotal;
	public Set<List<Integer>> visited;

	public Map<Integer, Integer> freq;
	public List<Integer> electedSampling;

	public Node root;

	public float time, timeSampling, timeScoring;
	public float timeCacheEval, timePruneEval, timeSelectNext, timeHeuristicEval, timeFork;

	public long begin;

	public Profile<Integer> profile;

	/**
	 * numZeroL - number of times invoking the heuristic but fails to find a
	 * loser that can be directly eliminated, numSingleL - number of times to
	 * invoke the heuristic and to eliminate one single loser, numMultiL -
	 * number of times to invoke the heuristic and eliminate multiple losers one
	 * time
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
	 * recursive or stack (or fringe) version
	 */
	public boolean recursive;

	public STV() {}

	public STV(boolean h, boolean c, boolean p, boolean s, boolean r) {
		this.heuristic = h;
		this.cache = c;
		this.pruning = p;
		this.sampling = s;
		this.recursive = r;
	}

	public class Node {
		public int priority = 0;
		public int[] scores;
		public List<Integer> state;

		public Node(List<Integer> candidates) {
			this.state = new ArrayList<>(candidates);
		}

		public Node(List<Integer> candidates, Integer eliminated) {
			this.state = new ArrayList<>(candidates);
			this.state.remove(eliminated);
		}

		int getPriority() {
			if (priority == 0) {
				priority = (numItemTotal - state.size()) * numItemTotal;
				for (int item : state) {
					if (freq.get(item) == 0)
						priority++;
				}
			}
			return priority;
		}
	}

	/**
	 * Calculate plurality score from a preference profile with those unstated
	 * candidates eliminated
	 * 
	 * @param profile
	 * @param candidates
	 * @return Plurality score of specific candidates
	 */
	public int[] getPluralityScore(Profile<Integer> profile, List<Integer> candidates) {
		// System.out.println("Scoring: " + candidates);

		long start = System.nanoTime();

		int[] votes = profile.votes;
		int[] scores = new int[candidates.size()];

		int c = 0;
		for (Integer[] pref : profile.data) {
			for (int i = 0; i < pref.length; i++) {
				int item = pref[i];
				int index = candidates.indexOf(item);
				// item has been eliminated
				if (index == -1)
					continue;
				scores[index] += votes[c];
				break;
			}
			c++;
		}
		timeScoring += (System.nanoTime() - start);
		numScoring++;
		return scores;
	}

	/**
	 * @param remaining
	 * @return Evaluate whether all remaining alternatives have been elected as
	 *         the winners
	 */
	boolean foundAll(List<Integer> remaining) {
		long start = System.nanoTime();
		for (int item : remaining) {
			if (freq.get(item) == 0) {
				timePruneEval += (System.nanoTime() - start);
				return false;
			}
		}
		timePruneEval += (System.nanoTime() - start);
		visited.add(remaining);
		return true;
	}

	/**
	 * According to the constraints on votes, we get all the candidates who will
	 * be eliminated no matter which tie-breaking rule to be used.
	 * 
	 * @param items
	 * @param scores
	 * @param tiers
	 * @return eliminated candidates
	 */
	List<Integer> getEliminatedCandidates(List<Integer> items, int[] scores, TreeMap<Integer, List<Integer>> tiers) {
		long start = System.nanoTime();

		tiers.clear();
		for (int i = 0; i < scores.length; i++) {
			int score = scores[i];
			List<Integer> member = tiers.get(score);
			if (member == null)
				member = new ArrayList<>();
			member.add(items.get(i));
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
	 * Eliminate alternatives directly and safely without considering the
	 * relative ordering to eliminate them
	 * 
	 * @param candidates
	 * @param scores
	 * @param tiers
	 * @return null if an alternative has been decleared as the winner, the
	 *         elimination is terminated immediately.
	 */
	int[] eliminateHSA(List<Integer> candidates, int[] scores, final TreeMap<Integer, List<Integer>> tiers) {
		List<Integer> losers = null;
		while ((losers = getEliminatedCandidates(candidates, scores, tiers)) != null) {
			long start = System.nanoTime();

			numNodeWH++;
			numNodeWOH += losers.size();

			if (losers.size() > 1)
				numMultiL++;
			else
				numSingleL++;

			visited.add(new ArrayList<>(candidates));
			candidates.removeAll(losers);

			if (candidates.size() == 1) {
				int item = candidates.get(0);
				freq.put(item, freq.get(item) + 1);
				timeHeuristicEval += (System.nanoTime() - start);
				visited.add(candidates);
				return null;
			}
			timeHeuristicEval += (System.nanoTime() - start);
			if (visited.contains(candidates))
				return null;

			scores = getPluralityScore(profile, candidates);
		}
		numFailH++;
		return scores;
	}

	/**
	 * Split with prunning
	 * 
	 * @param candidates
	 * @param tied
	 */
	void splitWPruning(List<Integer> candidates, List<Integer> tied) {
		long start = System.nanoTime();
		if (visited.contains(candidates)) {
			numCacheHit++;
			timeCacheEval += System.nanoTime() - start;
			return;
		} else {
			numCacheMiss++;
			timeCacheEval += (System.nanoTime() - start);
		}

		int numElim = tied.size();
		numNodeFull += numElim;

		Node child;
		List<Integer> items;

		while ((numElim = tied.size()) > 0) {
			start = System.nanoTime();
			int highest = -1, index = -1;
			// select the next alternative to eliminate
			for (int i = 0; i < numElim; i++) {
				int f = freq.get(tied.get(i));
				if (f > highest) {
					index = i;
					highest = f;
				}
			}

			int next = tied.remove(index);
			timeSelectNext += (System.nanoTime() - start);

			start = System.nanoTime();
			child = new Node(candidates, next);
			timeFork += (System.nanoTime() - start);

			start = System.nanoTime();
			items = new ArrayList<>(child.state);
			if (visited.contains(items)) {
				numCacheHit++;
				timeCacheEval += (System.nanoTime() - start);
				continue;
			} else {
				numCacheMiss++;
				timeCacheEval += (System.nanoTime() - start);
			}

			if (foundAll(items)) {
				numPruneHit++;
				continue;
			} else
				numPruneMiss++;

			elect(child);

			start = System.nanoTime();
			visited.add(items);
			timeCacheEval += (System.nanoTime() - start);

			numNode++;

			/**
			 * remaining candidates in parent's node have all been elected as
			 * the winners, no need to further split the parent node via
			 * elimination of the other tied candidates
			 **/
			if (foundAll(candidates)) {
				numPruneHit++;
				return;
			} else
				numPruneMiss++;
		}
	}

	/**
	 * Split node without pruning
	 * 
	 * @param candidates
	 * @param tied
	 */
	void splitWOPruning(List<Integer> candidates, List<Integer> tied) {
		int numElim = tied.size();
		numNodeFull += numElim;

		Node child = null;
		while ((numElim = tied.size()) > 0) {
			int highest = -1, index = -1;
			// select the next alternative to eliminate
			for (int i = 0; i < numElim; i++) {
				int f = freq.get(tied.get(i));
				if (f > highest) {
					index = i;
					highest = f;
				}
			}
			int next = tied.remove(index);
			child = new Node(candidates, next);
			elect(child);
		}
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
		int[] scores = getPluralityScore(profile, candidates);
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
	 * Recursion version
	 * 
	 * @param node
	 */
	void elect(Node node) {
		if (node.state.size() == 1) {
			int item = node.state.get(0);
			freq.put(item, freq.get(item) + 1);
			visited.add(node.state);
			return;
		}
		int[] scores = getPluralityScore(profile, node.state);

		TreeMap<Integer, List<Integer>> tiers = null;
		if (heuristic) {
			tiers = new TreeMap<>(Collections.reverseOrder());
			scores = eliminateHSA(node.state, scores, tiers);
			if (scores == null)
				return;
		} else {
			int[] max = MathLib.argmax(scores);
			if (scores[max[0]] > profile.numVoteTotal / 2) {
				int item = node.state.get(max[0]);
				freq.put(item, freq.get(item) + 1);
				visited.add(node.state);
				return;
			}
		}

		long start = System.nanoTime();
		List<Integer> tied = null;
		if (tiers == null) {
			int[] min = MathLib.argmin(scores);
			tied = new ArrayList<>();
			for (int i : min)
				tied.add(node.state.get(i));
		} else {
			int smallest = tiers.lastKey();
			tied = tiers.get(smallest);
		}
		timeSelectNext += (System.nanoTime() - start);

		if (cache)
			splitWPruning(node.state, tied);
		else
			splitWOPruning(node.state, tied);
	}

	/**
	 * Stack version
	 */
	void elect() {
		LinkedList<Node> fringe = new LinkedList<>();
		fringe.add(root);

		TreeMap<Integer, List<Integer>> tiers = null;
		Node top = null;
		while (!fringe.isEmpty()) {
			top = fringe.pollLast();
			List<Integer> candidates = top.state;
			int size = candidates.size();
			if (size == 1) {
				int item = candidates.get(0);
				freq.put(item, freq.get(item) + 1);
				visited.add(candidates);
				continue;
			}

			long start = System.nanoTime();
			if (cache) {
				if (visited.contains(candidates)) {
					numCacheHit++;
					timeCacheEval += System.nanoTime() - start;
					continue;
				} else {
					numCacheMiss++;
					timeCacheEval += System.nanoTime() - start;
				}
			}

			if (pruning) {
				if (foundAll(candidates)) {
					numPruneHit++;
					continue;
				} else
					numPruneMiss++;
			}
			int[] scores = getPluralityScore(profile, candidates);

			if (heuristic) {
				tiers = new TreeMap<>(Collections.reverseOrder());
				scores = eliminateHSA(candidates, scores, tiers);
				if (scores == null)
					continue;
			}

			// else {
			// int[] max = MathLib.argmax(scores);
			// if (scores[max[0]] > profile.numVoteTotal / 2) {
			// int item = candidates.get(max[0]);
			// freq.put(item, freq.get(item) + 1);
			// visited.add(candidates);
			// continue;
			// }
			// }

			start = System.nanoTime();
			List<Integer> tied = null;
			if (tiers == null) {
				int[] min = MathLib.argmin(scores);
				tied = new ArrayList<>();
				for (int i : min)
					tied.add(candidates.get(i));
			} else {
				int smallest = tiers.lastKey();
				tied = tiers.get(smallest);
			}
			timeSelectNext += System.nanoTime() - start;

			if (candidates.size() == 1) {
				int item = candidates.get(0);
				freq.put(item, freq.get(item) + 1);
				visited.add(candidates);
				continue;
			}

			/**
			 * number of candidates does't change, no need to check the cache
			 * and pruning conditions again
			 **/
			if (candidates.size() < size) {
				if (cache) {
					start = System.nanoTime();
					if (visited.contains(candidates)) {
						numCacheHit++;
						timeCacheEval += System.nanoTime() - start;
						continue;
					} else {
						numCacheMiss++;
						timeCacheEval += System.nanoTime() - start;
					}
				}

				if (pruning) {
					if (foundAll(candidates)) {
						numPruneHit++;
						continue;
					} else
						numPruneMiss++;
				}
			}

			int numElim = tied.size();
			numNodeFull += numElim;

			if (cache) {
				for (int item : tied) {
					Node child = new Node(candidates, item);
					start = System.nanoTime();
					if (!visited.contains(child.state)) {
						numCacheMiss++;
						numNode++;
						fringe.add(child);
					} else
						numCacheHit++;
					timeCacheEval += System.nanoTime() - start;
				}
				start = System.nanoTime();
				visited.add(candidates);
				timeCacheEval += System.nanoTime() - start;
			} else {
				numNode += numElim;
				for (int item : tied) {
					Node child = new Node(candidates, item);
					fringe.add(child);
				}
			}
		}
	}

	/**
	 * @param profile
	 * @return All possible winners with various tie-breaking rules
	 */
	public List<Integer> getAllWinners(Profile<Integer> profile) {
		visited = new HashSet<>();

		numFailH = numSingleL = numMultiL = 0;
		numNodeWH = numNodeWOH = 0;
		numNode = numNodeFull = 0;

		numCacheHit = numPruneHit = numScoring = 0;
		numCacheMiss = numPruneMiss = 0;

		Integer[] items = profile.getSortedItems();

		BruteForceSTV stvBF = null;
		if (sampling) {
			stvBF = new BruteForceSTV();
			int m = items.length;
			stvBF.getAllWinners(profile, m);
			freq = stvBF.countElected;
		} else {
			freq = new HashMap<>();
			for (Integer item : items)
				freq.put(item, 0);
		}

		root = new Node(Arrays.asList(items));

		timeScoring = 0;
		timeCacheEval = timePruneEval = timeSelectNext = timeHeuristicEval = timeFork;

		begin = System.nanoTime();
		this.profile = preprocess(profile, root.state);
		this.numItemTotal = root.state.size();
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
		// M40N70-39, M50N70-23, M30N30-237, M30N30-238
		TickClock.beginTick();

		String base = "/Users/chjiang/Documents/csc/";
		String dataset = "soc-3";

		Path input = Paths.get(base + dataset + "/M30N30-237.csv");
		boolean heuristic = false, cache = true, pruning = true, sampling = false, recursive = false;
		STV rule = new STV(heuristic, cache, pruning, sampling, recursive);

		Profile<Integer> profile = DataEngine.loadProfile(input);
		List<Integer> winners = rule.getAllWinners(profile);

		String format = "#h=%d, #h_single=%d, #node=%d, #score=%d, t=%f, t_score=%f, t_hash=%f, t_heur=%f, t_new=%f, t_next=%f, t_eva_elected=%f, t_sample=%f, winners=%s\n";
		System.out.printf(format, rule.numNodeWH, rule.numSingleL, rule.numNode, rule.numScoring, rule.time,
				rule.timeScoring, rule.timeCacheEval, rule.timeHeuristicEval, rule.timeFork,
				rule.timeSelectNext, rule.timePruneEval, rule.timeSampling, winners);

		TickClock.stopTick();
	}
}