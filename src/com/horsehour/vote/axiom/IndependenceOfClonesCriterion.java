package com.horsehour.vote.axiom;

/**
 * The independence of clones criterion measures an voting method's robustness
 * to strategic nomination. Nicolaus Tideman first formulated the criterion,
 * which states that the addition of a candidate identical to one already
 * present in an election will not cause the winner of the election to change.
 * <p>
 * In some systems, the introduction of a clone tends to divide support between
 * the similar candidates, worsening all their chances. In some other systems,
 * the presence of a clone tends to reduce support for dissimilar candidates,
 * improving the chances of one (or more) of the similar candidates. In yet
 * other systems, the introduction of clones does not significantly affect the
 * chances of similar candidates. There are further systems where the effect of
 * the introduction of clones depends on the distribution of other votes.
 * <p>
 * Elections methods that fail independence of clones can either be clone
 * negative (the addition of an identical candidate decreases a candidate's
 * chance of winning) or clone positive (the reverse). The Borda count is an
 * example of a clone positive method. Plurality is an example of a clone
 * negative method because of vote-splitting.
 * <p>
 * Instant-runoff voting, approval voting and range voting meet the independence
 * of clones criterion. Some election methods that comply with the Condorcet
 * criterion such as Ranked pairs and Schulze also meet independence of clones.
 * <p>
 * The Borda count, Minimax, two-round system, Bucklin voting and plurality fail
 * the independence of clones criterion.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:29:21 PM, Jul 31, 2016
 *
 */

public class IndependenceOfClonesCriterion {

}
