package com.horsehour.vote.rule.multiround;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

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
public class BruteForceCoombs extends BruteForceSTV {
	public BruteForceCoombs() {}

	public int[] scoring(Profile<Integer> profile, List<Integer> state) {
		int[] votes = profile.votes;
		int[] scores = new int[state.size()];
		int c = 0;
		for (Integer[] pref : profile.data) {
			for (int i = pref.length - 1; i >= 0; i--) {
				int item = pref[i];
				int index = state.indexOf(item);/** item has been eliminated **/
				if (index == -1)
					continue;
				scores[index] -= votes[c];
				break;
			}
			c++;
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

		String base = "/users/chjiang/documents/csc/", dataset = "soc-3", file = "M10N10-3.csv";
		Profile<Integer> profile = DataEngine.loadProfile(Paths.get(base + dataset + "/" + file));

		BruteForceCoombs rule = new BruteForceCoombs();
		System.out.println(rule.getAllWinners(profile, 30));

		TickClock.stopTick();
	}
}
