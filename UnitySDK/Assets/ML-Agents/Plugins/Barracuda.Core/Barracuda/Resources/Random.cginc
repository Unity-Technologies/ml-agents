
// Based on: https://stackoverflow.com/questions/5149544/can-i-generate-a-random-number-inside-a-pixel-shader
// Output: Random number: [0,1), that is between 0.0 and 0.999999... inclusive.
// Author: Michael Pohoreski
// Copyright: Copyleft 2012 :-)
float RandomUsingCos(float4 seed)
{
	float4 K1 = float4(		// Transcendental numbers:
		0.64341054629,     	// (Cahen's constant)
		23.14069263277926,	// e^pi (Gelfond's constant)
		2.665144142690225,	// 2^sqrt(2) (Gelfond-Schneider constant)
		3.14159265359		// pi
	);
	return frac(cos(dot(seed, K1)) * 12345.6789);
}

// Based on: https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
// Author: Spatial
// 05 July 2013

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
uint hash(uint x)
{
	x += ( x << 10u );
	x ^= ( x >>  6u );
	x += ( x <<  3u );
	x ^= ( x >> 11u );
	x += ( x << 15u );
	return x;
}
uint hash( uint2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uint3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uint4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct(uint m)
{
	const uint ieeeMantissa = 0x007FFFFFu;	// binary32 mantissa bitmask
	const uint ieeeOne      = 0x3F800000u;	// 1.0 in IEEE binary32

	m &= ieeeMantissa;						// Keep only mantissa bits (fractional part)
	m |= ieeeOne;							// Add fractional part to 1.0

	float  f = asfloat(m);					// Range [1:2]
	return f - 1.0;							// Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
float RandomUsingHash(float4 seed)
{
	return floatConstruct(hash(asuint(seed)));
}


// More alternatives:
// https://github.com/ashima/webgl-noise
// https://www.shadertoy.com/view/4djSRW

// ------------------------------------------------------------------------------------------

float Random(float4 seed)
{
	return RandomUsingCos(seed);
}

float Bernoulli(float4 seed, float p)
{
	return Random(seed) <= p ? 1: 0;
}
