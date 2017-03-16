package com.horsehour.vote.rule.multiseat;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.DataEngine;
import com.horsehour.vote.Profile;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 11:06:40 PM, Jan 1, 2017
 */
public class Coombs extends STVPlus2 {
	public int n;

	public Coombs(boolean h, boolean c, boolean p, boolean s, boolean r, int pf) {
		super(h, c, p, s, r, pf);
	}

	void elect() {
		LinkedList<Node> fringe = new LinkedList<>();
		fringe.add(root);
		trace.put(numNode, numWinner);

		Node next = null;
		while (!fringe.isEmpty()) {
			if (pFunction > 0) {
				for (Node node : fringe)
					updatePriority(node);
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
				List<Integer> losers = getHeuristicLoser(state, scores);
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
					int[] min = MathLib.argmin(scores);
					for (int i : min)
						children.add(new Node(state, state.get(i)));
					timeFork += System.nanoTime() - start;
				}
			} else {
				start = System.nanoTime();
				int[] min = MathLib.argmin(scores);
				numNodeFull += min.length;
				// generate children based on their plurality scores
				for (int i : min)
					children.add(new Node(state, state.get(i)));
				timeFork += System.nanoTime() - start;
			}

			// add all children to fringe
			fringe.addAll(children);

			if (cache)
				visited.add(state);
			numNode++;
			trace.put(numNode, numWinner);
		}
	}

	void elect(Node node) {
		List<Integer> state = node.state;
		if (isUniqueRemaining(state) || (cache && hasVisited(state)) || (pruning && canPrune(state)))
			return;

		int[][] scores = scoringFirstAndLast(profile, state);
		int[] maximum = MathLib.argmax(scores[0]);

		if (containsUniqueWinner(state, scores[0]))
			return;

		maximum = MathLib.argmax(scores[1]);
		long start = System.nanoTime();
		List<Integer> tied = new ArrayList<>();
		for (int i = 0; i < maximum.length; i++)
			tied.add(state.get(maximum[i]));
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

			if (cache) {
				start = System.nanoTime();
				numNode++;
				visited.add(child.state);
				timeCacheEval += System.nanoTime() - start;
			}

			if (pruning && canPrune(state))
				continue;
		}

		if (cache) {
			numNode++;
			visited.add(state);
		}
	}

	List<Integer> getHeuristicLoser(List<Integer> state, int[] scores) {
		int[] rank = MathLib.getRank(scores, false);
		long start = System.nanoTime();
		int d = -n / 2;

		List<Integer> losers = new ArrayList<>();
		int i = 0;
		for (; i < scores.length; i++) {
			int score = scores[rank[i]];
			if (score < d)
				losers.add(state.get(rank[i]));
		}
		timeHeuristicEval += System.nanoTime() - start;

		if (losers.isEmpty())
			return null;
		else
			return losers;
	}

	public int[] scoring(Profile<Integer> profile, List<Integer> state) {
		numScoring++;
		long start = System.nanoTime();

		int[] votes = profile.votes;
		int[] scores = new int[state.size()];
		int c = 0;
		for (Integer[] pref : profile.data) {
			for (int i = pref.length - 1; i >= 0; i--) {
				int item = pref[i];
				int index = state.indexOf(item);
				/** item has been eliminated **/
				if (index == -1)
					continue;

				scores[index] -= votes[c];
				break;
			}
			c++;
		}
		timeScoring += System.nanoTime() - start;
		return scores;
	}

	/**
	 * @param profile
	 * @param state
	 * @return scores in terms of both first and last choices for each candidate
	 *         in state
	 */
	int[][] scoringFirstAndLast(Profile<Integer> profile, List<Integer> state) {
		numScoring++;
		long start = System.nanoTime();

		int[] votes = profile.votes;
		int[][] scores = new int[2][state.size()];
		int c = 0;
		for (Integer[] pref : profile.data) {
			for (int i = 0; i < pref.length; i++) {
				int item = pref[i];
				int index = state.indexOf(item);
				/** item has been eliminated **/
				if (index == -1)
					continue;
				scores[0][index] += votes[c];
				break;
			}

			for (int i = pref.length - 1; i >= 0; i--) {
				int item = pref[i];
				int index = state.indexOf(item);
				/** item has been eliminated **/
				if (index == -1)
					continue;
				scores[1][index] += votes[c];
				break;
			}
			c++;
		}
		timeScoring += System.nanoTime() - start;
		return scores;
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

		this.profile = profile;
		List<Integer> state = new ArrayList<>(Arrays.asList(items));
		freq = new HashMap<>();
		BruteForceCoombs ruleBF = null;
		if (sampling) {
			ruleBF = new BruteForceCoombs();
			List<Integer> winners = ruleBF.getAllWinners(profile, state.size());
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
		this.n = profile.numVoteTotal;

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

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/Users/chjiang/Documents/csc/";
		String dataset = "soc-3";

		boolean heuristic = false, cache = true, pruning = true;
		boolean sampling = false, recursive = false;
		int pf = 2;
		Coombs rule = new Coombs(heuristic, cache, pruning, sampling, recursive, pf);

		Path input = Paths.get(base + dataset + "/M10N10-3.csv");
		Profile<Integer> profile = DataEngine.loadProfile(input);
		List<Integer> winners = rule.getAllWinners(profile);

		String format = "#node=%d, #score=%d, #cache=%d, #success=%d, t=%f, t_score=%f, winners=%s\n";
		System.out.printf(format, rule.numNode, rule.numScoring, rule.visited.size(), rule.numNodeWH - rule.numSingleL,
				rule.time, rule.timeScoring, winners);

		TickClock.stopTick();
	}
}