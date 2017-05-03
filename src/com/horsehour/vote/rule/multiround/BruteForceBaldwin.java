package com.horsehour.vote.rule.multiround;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
import com.horsehour.vote.Profile;
import com.horsehour.vote.data.DataEngine;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 11:26:24 PM, Feb 15, 2017
 *
 */
public class BruteForceBaldwin extends BruteForceSTV {
	public Map<Integer, Map<Integer, Integer>> prefMatrix;

	public BruteForceBaldwin() {}

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

	void elect(List<Integer> remaining) {
		if (remaining.size() == 1) {
			int item = remaining.get(0);
			elected.add(item);
			return;
		}

		int[] scores = scoring(profile, remaining);
		List<Integer> tied = new ArrayList<>();
		int[] min = MathLib.argmin(scores);
		for (int i : min)
			tied.add(remaining.get(i));

		Integer eliminated = getEliminated(tied);
		remaining.remove(eliminated);
		elect(remaining);
	}

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "/users/chjiang/documents/csc/", dataset = "soc-3", file = "M20N20-2.csv";
		Profile<Integer> profile = DataEngine.loadProfile(Paths.get(base + dataset + "/" + file));

		BruteForceBaldwin rule = new BruteForceBaldwin();
		System.out.println(rule.getAllWinners(profile, 1000));

		TickClock.stopTick();
	}
}
