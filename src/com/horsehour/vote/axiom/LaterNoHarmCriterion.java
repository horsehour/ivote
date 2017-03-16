package com.horsehour.vote.axiom;

/**
 * The later-no-harm criterion is a voting system criterion formulated by
 * Douglas Woodall. The criterion is satisfied if, in any election, a voter
 * giving an additional ranking or positive rating to a less preferred candidate
 * cannot cause a more preferred candidate to lose.
 * <p>
 * Single transferable vote (including Instant Runoff Voting and Contingent
 * vote), Minimax Condorcet (pairwise opposition variant which does not satisfy
 * the Condorcet Criterion), and Descending Solid Coalitions, a variant of
 * Woodall's Descending Acquiescing Coalitions rule, satisfy the later-no-harm
 * criterion.
 * <p>
 * However, if a method permits incomplete ranking of candidates, and if a
 * majority of initial round votes is required for election, it cannot satisfy
 * Later-no-harm, because a lower preference vote cast may create a majority for
 * that lower preference, whereas if the vote was not cast, the election could
 * fail, proceed to a runoff, repeated ballot or other process, and the favored
 * candidate could possibly win.
 * <p>
 * Approval voting, Borda count, Range voting, Schulze method and Bucklin voting
 * do not satisfy later-no-harm. The Condorcet criterion is incompatible with
 * later-no-harm.
 * <p>
 * When Plurality is being used to fill two or more seats in a single district
 * (Plurality-at-large) it fails later-no-harm.
 * <p>
 * The later-no-harm criterion is by definition inapplicable to any voting
 * system in which a voter is not allowed to express more than one choice,
 * including plurality voting, the system most commonly used in Canada, India,
 * the UK, and the USA.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:31:29 PM, Jul 31, 2016
 *
 */

public class LaterNoHarmCriterion {

}
