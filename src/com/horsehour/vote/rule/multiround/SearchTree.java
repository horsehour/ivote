package com.horsehour.vote.rule.multiround;

import java.util.function.Function;

import com.horsehour.vote.rule.multiround.BoostSTV.Node;

/***
 * Search tree for an election
 * 
 * @author Chunheng Jiang
 * @since May 4, 2017
 * @version 1.0
 */
public class SearchTree {
	static Function<Integer, String> space = d -> {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < d; i++)
			sb.append("  ");
		return sb.toString();
	};

	/***
	 * Construct the search tree from a root node
	 * 
	 * @param node
	 * @return search tree
	 */
	public static String create(BoostSTV.Node node) {
		String name = node.order + " : " + node.state.toString().replaceAll("\\[|\\]| ", "");
		if (node.state.size() > 1 && node.winnersPredicted != null)
			name += node.winnersPredicted;
		name = "\"name\": \"" + name + "\"";

		StringBuffer sb = new StringBuffer();
		if (node.leaf)
			sb.append(space.apply(node.depth)).append("{").append(name).append("}");
		else {
			int nc = node.children.size();
			sb.append(space.apply(node.depth)).append("{\n");
			sb.append(space.apply(node.depth + 1)).append(name).append(", \n");
			sb.append(space.apply(node.depth + 1)).append("\"children\": [\n");
			for (int i = 0; i < nc; i++) {
				Node child = node.children.get(i);
				sb.append(create(child));
				if (i < nc - 1)
					sb.append(",");
				sb.append("\n");
			}
			sb.append(space.apply(node.depth + 1)).append("]\n");
			sb.append(space.apply(node.depth)).append("}");
		}
		return sb.toString();
	}
}