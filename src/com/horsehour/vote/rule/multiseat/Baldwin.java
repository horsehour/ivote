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
import java.util.Map;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.DataEngine;
import com.horsehour.vote.Profile;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 10:37:43 PM, Jan 1, 2017
 *
 */

public class Baldwin extends STVPlus2 {
	public Map<Integer, Map<Integer, Integer>> prefMatrix;
	public int n;

	public Baldwin(boolean h, boolean c, boolean p, boolean s, boolean r, int pf) {
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

	List<Integer> getHeuristicLoser(List<Integer> state, int[] scores) {
		int[] rank = MathLib.getRank(scores, false);
		long start = System.nanoTime();
		int m = state.size();
		int d = (m - 1) * (m - 2) * n / 2;

		int i = 0;
		int remaining = MathLib.Data.sum(scores);
		for (; i < scores.length; i++) {
			int score = scores[rank[i]];
			remaining -= score;
			if (score + d > remaining)
				break;
			d -= (m - 2 - i);
		}

		if (i == m - 1) {
			timeHeuristicEval += System.nanoTime() - start;
			return null;
		}

		i += 1;
		List<Integer> losers = new ArrayList<>();
		for (; i < m; i++)
			losers.add(state.get(rank[i]));
		timeHeuristicEval += System.nanoTime() - start;
		return losers;
	}

	public void getPrefMatrix(Profile<Integer> profile) {
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

	public int[] scoring(Profile<Integer> profile, List<Integer> state) {
		if (prefMatrix == null)
			getPrefMatrix(profile);

		numScoring++;
		long start = System.nanoTime();

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
		this.prefMatrix = null;
		List<Integer> state = Arrays.asList(items);

		freq = new HashMap<>();
		BruteForceBaldwin ruleBF = null;
		if (sampling) {
			ruleBF = new BruteForceBaldwin();
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
		this.numItemTotal = items.length;
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
		String dataset = "soc-3-baldwin";

		boolean heuristic = false, cache = false, pruning = false;
		boolean sampling = false, recursive = false;
		int pf = 0;
		
		Baldwin rule = new Baldwin(heuristic, cache, pruning, sampling, recursive, pf);

		Path input = Paths.get(base + dataset + "/M10N90-999.csv");
		Profile<Integer> profile = DataEngine.loadProfile(input);
		List<Integer> winners = rule.getAllWinners(profile);

		String format = "#node=%d, #score=%d, #cache=%d, t=%f, t_score=%f, winners=%s\n";
		System.out.printf(format, rule.numNode, rule.numScoring, rule.visited.size(), rule.time, rule.timeScoring,
				winners);

		TickClock.stopTick();
	}
}
