package com.horsehour.vote.rule.multiround;

import java.util.Map;

import com.horsehour.vote.Profile;

/**
 * @author Chunheng Jiang
 * @since Mar 27, 2017
 * @version 1.0
 */
public class PriorityFunction {
	public int id = 0;

	public static void computePF1(BoostSTV.Node node, Profile<Integer> profile, Map<Integer, Integer> winFreq,
			int numItemTotal) {
		int m = node.state.size();
		// if (node.pred == null)
		// node.pred = predict(profile, node.state);

		node.priority = numItemTotal - m;
		double expectation = 0;
		for (int i = 0; i < m; i++)
			if (winFreq.get(node.state.get(i)) == 0 && node.pred.get(i) > 0.5)
				expectation += node.pred.get(i);
		node.priority *= expectation;
	}
}
