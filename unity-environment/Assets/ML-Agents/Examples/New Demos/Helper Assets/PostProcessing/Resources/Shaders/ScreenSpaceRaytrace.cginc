/**
\author Michael Mara and Morgan McGuire, Casual Effects. 2015.
*/

#ifndef __SCREEN_SPACE_RAYTRACE__
#define __SCREEN_SPACE_RAYTRACE__

sampler2D_float _CameraDepthTexture;

float distanceSquared(float2 A, float2 B)
{
    A -= B;
    return dot(A, A);
}

float distanceSquared(float3 A, float3 B)
{
    A -= B;
    return dot(A, A);
}

void swap(inout float v0, inout float v1)
{
    float temp = v0;
    v0 = v1;
    v1 = temp;
}

bool isIntersecting(float rayZMin, float rayZMax, float sceneZ, float layerThickness)
{
    return (rayZMax >= sceneZ - layerThickness) && (rayZMin <= sceneZ);
}

void rayIterations(in bool traceBehindObjects, inout float2 P, inout float stepDirection, inout float end, inout int stepCount, inout int maxSteps, inout bool intersecting,
        inout float sceneZ, inout float2 dP, inout float3 Q, inout float3 dQ, inout float k, inout float dk,
        inout float rayZMin, inout float rayZMax, inout float prevZMaxEstimate, inout bool permute, inout float2 hitPixel,
        inout float2 invSize, inout float layerThickness)
{
    bool stop = intersecting;

    UNITY_LOOP
    for (; (P.x * stepDirection) <= end && stepCount < maxSteps && !stop; P += dP, Q.z += dQ.z, k += dk, stepCount += 1)
    {
        // The depth range that the ray covers within this loop iteration.
        // Assume that the ray is moving in increasing z and swap if backwards.
        rayZMin = prevZMaxEstimate;
        //rayZMin = (dQ.z * -0.5 + Q.z) / (dk * -0.5 + k);
        // Compute the value at 1/2 pixel into the future
        rayZMax = (dQ.z * 0.5 + Q.z) / (dk * 0.5 + k);
        prevZMaxEstimate = rayZMax;

        if (rayZMin > rayZMax)
        {
            swap(rayZMin, rayZMax);
        }

        // Undo the homogeneous operation to obtain the camera-space
        // Q at each point
        hitPixel = permute ? P.yx : P;

        sceneZ = tex2Dlod(_CameraDepthTexture, float4(hitPixel * invSize,0,0)).r;
        sceneZ = -LinearEyeDepth(sceneZ);

        bool isBehind = (rayZMin <= sceneZ);
        intersecting = isBehind && (rayZMax >= sceneZ - layerThickness);
        stop = traceBehindObjects ? intersecting : isBehind;

    } // pixel on ray

    P -= dP, Q.z -= dQ.z, k -= dk;
}

/**
  \param csOrigin must have z < -0.01, and project within the valid screen rectangle
  \param stepRate Set to 1.0 by default, higher to step faster
 */
bool castDenseScreenSpaceRay
   (float3          csOrigin,
    float3          csDirection,
    float4x4        projectToPixelMatrix,
    float2          csZBufferSize,
    float3          clipInfo,
    float           jitterFraction,
    int             maxSteps,
    float           layerThickness,
    float           maxRayTraceDistance,
    out float2      hitPixel,
    int             stepRate,
    bool            traceBehindObjects,
    out float3      csHitPoint,
    out float       stepCount) {

    float2 invSize = float2(1.0 / csZBufferSize.x, 1.0 / csZBufferSize.y);

    // Initialize to off screen
    hitPixel = float2(-1, -1);

    float nearPlaneZ = -0.01;
    // Clip ray to a near plane in 3D (doesn't have to be *the* near plane, although that would be a good idea)
    float rayLength = ((csOrigin.z + csDirection.z * maxRayTraceDistance) > nearPlaneZ) ?
                        ((nearPlaneZ - csOrigin.z) / csDirection.z) :
                        maxRayTraceDistance;

    float3 csEndPoint = csDirection * rayLength + csOrigin;

    // Project into screen space
    // This matrix has a lot of zeroes in it. We could expand
    // out these multiplies to avoid multiplying by zero
    // ...but 16 MADDs are not a big deal compared to what's ahead
    float4 H0 = mul(projectToPixelMatrix, float4(csOrigin, 1.0));
    float4 H1 = mul(projectToPixelMatrix, float4(csEndPoint, 1.0));

    // There are a lot of divisions by w that can be turned into multiplications
    // at some minor precision loss...and we need to interpolate these 1/w values
    // anyway.
    //
    // Because the caller was required to clip to the near plane,
    // this homogeneous division (projecting from 4D to 2D) is guaranteed
    // to succeed.
    float k0 = 1.0 / H0.w;
    float k1 = 1.0 / H1.w;

    // Screen-space endpoints
    float2 P0 = H0.xy * k0;
    float2 P1 = H1.xy * k1;

    // Switch the original points to values that interpolate linearly in 2D:
    float3 Q0 = csOrigin * k0;
    float3 Q1 = csEndPoint * k1;

#if 1 // Clipping to the screen coordinates. We could simply modify maxSteps instead
    float yMax = csZBufferSize.y - 0.5;
    float yMin = 0.5;
    float xMax = csZBufferSize.x - 0.5;
    float xMin = 0.5;

    // 2D interpolation parameter
    float alpha = 0.0;
    // P0 must be in bounds
    if (P1.y > yMax || P1.y < yMin) {
        float yClip = (P1.y > yMax) ? yMax : yMin;
        float yAlpha = (P1.y - yClip) / (P1.y - P0.y); // Denominator is not zero, since P0 != P1 (or P0 would have been clipped!)
        alpha = yAlpha;
    }

    // P0 must be in bounds
    if (P1.x > xMax || P1.x < xMin) {
        float xClip = (P1.x > xMax) ? xMax : xMin;
        float xAlpha = (P1.x - xClip) / (P1.x - P0.x); // Denominator is not zero, since P0 != P1 (or P0 would have been clipped!)
        alpha = max(alpha, xAlpha);
    }

    // These are all in homogeneous space, so they interpolate linearly
    P1 = lerp(P1, P0, alpha);
    k1 = lerp(k1, k0, alpha);
    Q1 = lerp(Q1, Q0, alpha);
#endif

    // We're doing this to avoid divide by zero (rays exactly parallel to an eye ray)
    P1 = (distanceSquared(P0, P1) < 0.0001) ? P0 + float2(0.01, 0.01) : P1;

    float2 delta = P1 - P0;

    // Assume horizontal
    bool permute = false;
    if (abs(delta.x) < abs(delta.y)) {
        // More-vertical line. Create a permutation that swaps x and y in the output
        permute = true;

        // Directly swizzle the inputs
        delta = delta.yx;
        P1 = P1.yx;
        P0 = P0.yx;
    }

    // From now on, "x" is the primary iteration direction and "y" is the secondary one

    float stepDirection = sign(delta.x);
    float invdx = stepDirection / delta.x;
    float2 dP = float2(stepDirection, invdx * delta.y);

    // Track the derivatives of Q and k
    float3 dQ = (Q1 - Q0) * invdx;
    float   dk = (k1 - k0) * invdx;

    dP *= stepRate;
    dQ *= stepRate;
    dk *= stepRate;

    P0 += dP * jitterFraction;
    Q0 += dQ * jitterFraction;
    k0 += dk * jitterFraction;

    // Slide P from P0 to P1, (now-homogeneous) Q from Q0 to Q1, and k from k0 to k1
    float3 Q = Q0;
    float  k = k0;

    // We track the ray depth at +/- 1/2 pixel to treat pixels as clip-space solid
    // voxels. Because the depth at -1/2 for a given pixel will be the same as at
    // +1/2 for the previous iteration, we actually only have to compute one value
    // per iteration.
    float prevZMaxEstimate = csOrigin.z;
    stepCount = 0.0;
    float rayZMax = prevZMaxEstimate, rayZMin = prevZMaxEstimate;
    float sceneZ = 100000;

    // P1.x is never modified after this point, so pre-scale it by
    // the step direction for a signed comparison
    float end = P1.x * stepDirection;

    bool intersecting = isIntersecting(rayZMin, rayZMax, sceneZ, layerThickness);
    // We only advance the z field of Q in the inner loop, since
    // Q.xy is never used until after the loop terminates

    //int rayIterations = min(maxSteps, stepsToGetOffscreen);


    float2 P = P0;

    int originalStepCount = 0;
    rayIterations(traceBehindObjects, P, stepDirection, end,  originalStepCount,  maxSteps, intersecting,
         sceneZ, dP, Q, dQ,  k,  dk,
         rayZMin,  rayZMax,  prevZMaxEstimate, permute, hitPixel,
         invSize,  layerThickness);


    stepCount = originalStepCount;

    // Loop only advanced the Z component. Now that we know where we are going
    // update xy
    Q.xy += dQ.xy * stepCount;
    // Q is a vector, so we are trying to get by with 1 division instead of 3.
    csHitPoint = Q * (1.0 / k);

    return intersecting;
}

#endif // __SCREEN_SPACE_RAYTRACE__
