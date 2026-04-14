**This is a project for my mathematical modeling class**
            *Handwriten numeral recognition* 

*Idea of the project*
Given a grayscale image of a number 0-9 you need to identify which of theese numbers it is.

We do this by building a matrix lets call it **Ai** where i={0-9} and in this matrix each column represents a an image of number *i* flattened to a vector of length *k^2*.

We then recieve an image, flatten it to a vector lets call it *b* also of length *k^2* and we then search for the best possible solution of the problem. 
That solution is the one with the smallest norm or the best solution to the equation 
*xi=Ai^+b*
We calculate that equation for each *i* and choose the one where *||b-Aixi||* is smallest.

The project includes:
 - Training data gotten from:
   - num-rec.nmiklavcic.com
   - Paper scans of numbers gotten from a variety of people
 - A pre-processing algorythm that finds the center of numbers and tries to match them up as best as possible ()
 - 

<computation help>
Efficient Computation via SVD
Direct pseudoinverse is expensive. Instead, precompute the SVD of each Aᵢ:

Aᵢ = Uᵢ Sᵢ Vᵢᵀ
Then the least-squares solution becomes:

xᵢ = Aᵢ⁺ b = Vᵢ Sᵢ⁻¹ Uᵢᵀ b
And the projection of b onto the column space of Aᵢ is:

Aᵢ xᵢ = Uᵢ Uᵢᵀ b
So the residual is simply ‖b - Uᵢ Uᵢᵀ b‖ — just the component of b orthogonal to the column space of Aᵢ. You only need Uᵢ at test time (you can discard Sᵢ and Vᵢ after SVD, or keep a truncated Uᵢ for efficiency).
</computation help>

<preprocessing help>
Load each png
resize to k×k size (lets use 28×28 as that is the standard for this problem)
normalize pixel values (0-1 float)
flatten to k^2 vector
stack vectors in to Ai
Save the matrices for use during recognition

</preprocessing help>