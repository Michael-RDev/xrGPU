// SPDX-License-Identifier: Apache-2.0
// Frozen original native numerical helpers, independent of legacy dispatch.
// SPDX-License-Identifier: Apache-2.0
// Original native scalar kernels shared by packed Vulkan dispatch variants.
// Math-library precision and device float controls still need GPU validation.
float unaryNaN() { return uintBitsToFloat(0x7fc00000u); }
float unaryInfinity() { return uintBitsToFloat(0x7f800000u); }
float unarySignedZero(float x) {
  return uintBitsToFloat(floatBitsToUint(x) & 0x80000000u);
}

// Original double-float arithmetic keeps the range-reduction residual and
// polynomial error below one float32 rounding unit. No shaderFloat64 needed.
vec2 unaryDFNormalize(float a,float b) {
  precise float s=a+b;
  return vec2(s,b-(s-a));
}
vec2 unaryDFAdd(vec2 a,vec2 b) {
  precise float s=a.x+b.x;
  precise float v=s-a.x;
  precise float error=((a.x-(s-v))+(b.x-v))+a.y+b.y;
  return unaryDFNormalize(s,error);
}
vec2 unaryDFMul(vec2 a,vec2 b) {
  precise float p=a.x*b.x;
  precise float error=fma(a.x,b.x,-p)+a.x*b.y+a.y*b.x;
  return unaryDFNormalize(p,error);
}
vec2 unaryDFDiv(vec2 a,vec2 b) {
  precise float q=a.x/b.x;
  vec2 product=unaryDFMul(b,vec2(q,0.0));
  vec2 remainder=unaryDFAdd(a,vec2(-product.x,-product.y));
  return unaryDFNormalize(q,(remainder.x+remainder.y)/b.x);
}
vec2 unaryLn2() { return vec2(0.693147182464599609f,-1.90465421212593355e-09f); }

float unaryScalePowerOfTwo(vec2 value,int power) {
  uint bits=floatBitsToUint(value.x);
  uint signbit=bits&0x80000000u;
  if ((bits&0x7fffffffu)==0u) return value.x;
  int exponent=int((bits>>23u)&255u)+power;
  if (exponent>=255) return uintBitsToFloat(signbit|0x7f800000u);
  if (exponent>0) return uintBitsToFloat(signbit|(uint(exponent)<<23u)|(bits&0x007fffffu));
  int shift=1-exponent;
  if (shift>24) return uintBitsToFloat(signbit);
  uint significand=(bits&0x007fffffu)|0x00800000u;
  uint rounded=significand>>uint(shift);
  uint lost=significand&((1u<<uint(shift))-1u);
  uint halfway=1u<<uint(shift-1);
  float residual=signbit==0u?value.y:-value.y;
  if (lost>halfway || (lost==halfway &&
      (residual>0.0 || (residual==0.0 && (rounded&1u)!=0u)))) ++rounded;
  return uintBitsToFloat(signbit|rounded);
}
struct UnaryExponential { vec2 value; int power; };
UnaryExponential unaryExpReduced(float x) {
  float k=roundEven(x*1.4426950408889634f);
  vec2 reduction=unaryDFAdd(vec2(x,0.0),unaryDFMul(vec2(-k,0.0),unaryLn2()));
  vec2 p=vec2(1.60590443720742826e-10f,-5.35252651156272563e-18f);
  p=unaryDFAdd(vec2(2.08767558795841524e-09f,1.10828391474598523e-16f),unaryDFMul(reduction,p));
  p=unaryDFAdd(vec2(2.50521079436794025e-08f,4.41762304464836646e-16f),unaryDFMul(reduction,p));
  p=unaryDFAdd(vec2(2.75573199814971304e-07f,-7.57511220905119492e-15f),unaryDFMul(reduction,p));
  p=unaryDFAdd(vec2(2.75573188446287531e-06f,3.79357122429722915e-14f),unaryDFMul(reduction,p));
  p=unaryDFAdd(vec2(2.48015876422869042e-05f,-3.40699609366681977e-13f),unaryDFMul(reduction,p));
  p=unaryDFAdd(vec2(0.000198412701138295233f,-2.72559687493345582e-12f),unaryDFMul(reduction,p));
  p=unaryDFAdd(vec2(0.00138888892251998186f,-3.3631094437103215e-11f),unaryDFMul(reduction,p));
  p=unaryDFAdd(vec2(0.00833333376795053482f,-4.34617203337595015e-10f),unaryDFMul(reduction,p));
  p=unaryDFAdd(vec2(0.0416666679084300995f,-1.24176346982807218e-09f),unaryDFMul(reduction,p));
  p=unaryDFAdd(vec2(0.166666671633720398f,-4.9670538793122887e-09f),unaryDFMul(reduction,p));
  p=unaryDFAdd(vec2(0.5f,0.0f),unaryDFMul(reduction,p));
  p=unaryDFAdd(vec2(1.0f,0.0f),unaryDFMul(reduction,p));
  p=unaryDFAdd(vec2(1.0f,0.0f),unaryDFMul(reduction,p));

  UnaryExponential result;
  result.value=p;
  result.power=int(k);
  return result;
}
float unaryExpMul(float x,float factor) {
  if (isnan(x)) return x;
  if (x>100.0) return factor<0.0?-unaryInfinity():unaryInfinity();
  if (x<-150.0) return unarySignedZero(factor);
  UnaryExponential parts=unaryExpReduced(x);
  return unaryScalePowerOfTwo(unaryDFMul(parts.value,vec2(factor,0.0)),parts.power);
}
// Activation callers bound x to [-20,0], so both scaled components remain
// normal. Keeping their residual avoids rounding exp before the final result.
vec2 unaryExpDF(float x) {
  UnaryExponential parts=unaryExpReduced(x);
  float high=unaryScalePowerOfTwo(vec2(parts.value.x,0.0),parts.power);
  float low=unaryScalePowerOfTwo(vec2(parts.value.y,0.0),parts.power);
  return unaryDFNormalize(high,low);
}
float unaryExp(float x) { return unaryExpMul(x,1.0); }
vec2 unaryTwiceAtanhDF(vec2 t) {
  vec2 z=unaryDFMul(t,t);
  vec2 p=vec2(0.0322580635547637939f,9.61365209661835252e-10f);
  p=unaryDFAdd(vec2(0.034482758492231369f,1.28458285675314698e-10f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(0.0370370373129844666f,-2.75947431571665902e-10f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(0.0399999991059303284f,8.94069651646844932e-10f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(0.0434782616794109344f,-8.09845690508126381e-10f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(0.0476190485060214996f,-8.86973883229558169e-10f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(0.0526315793395042419f,-3.92135823812367335e-10f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(0.0588235296308994293f,-2.19134724255276581e-10f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(0.0666666701436042786f,-3.47693762670076012e-09f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(0.076923079788684845f,-2.86560797313484272e-09f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(0.0909090936183929443f,-2.70930211598852111e-09f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(0.1111111119389534f,-8.27842294714997706e-10f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(0.142857149243354797f,-6.3862120036617398e-09f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(0.200000002980232239f,-2.98023228317845224e-09f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(0.333333343267440796f,-9.9341077586245774e-09f),unaryDFMul(z,p));
  p=unaryDFAdd(vec2(1.0f,0.0f),unaryDFMul(z,p));

  return unaryDFMul(unaryDFMul(vec2(2.0,0.0),t),p);
 }
vec2 unaryLogMantissa(vec2 m) {
  return unaryTwiceAtanhDF(unaryDFDiv(unaryDFAdd(m,vec2(-1.0,0.0)),
      unaryDFAdd(m,vec2(1.0,0.0))));
}
vec2 unaryLogOnePlusDF(vec2 t) {
  // log1p(t)=2*atanh(t/(2+t)); form the small numerator directly so
  // its low residual is never lost by first rounding 1+t.
  return unaryTwiceAtanhDF(unaryDFDiv(t,unaryDFAdd(vec2(2.0,0.0),t)));
}
float unaryLogBits(uint bits,int exponentOffset) {
  uint magnitude=bits&0x7fffffffu;
  if (magnitude==0u) return -unaryInfinity();
  if ((bits&0x80000000u)!=0u) return unaryNaN();
  if (magnitude>=0x7f800000u) return uintBitsToFloat(bits);
  int exponent=int(magnitude>>23u)-127;
  uint fraction=magnitude&0x007fffffu;
  if ((magnitude>>23u)==0u) {
    uint shift=0u;
    while ((fraction&0x00800000u)==0u) { fraction<<=1u;++shift; }
    fraction&=0x007fffffu;
    exponent=-126-int(shift);
  }
  float mantissa=uintBitsToFloat(0x3f800000u|fraction);
  if (mantissa>1.4142135623730951f) {mantissa*=0.5;++exponent;}
  vec2 logarithm=unaryLogMantissa(vec2(mantissa,0.0));
  vec2 result=unaryDFAdd(logarithm,unaryDFMul(vec2(float(exponent+exponentOffset),0.0),unaryLn2()));
  return result.x;
}
float unaryLogOnePlus(float t) {
  vec2 m=unaryDFAdd(vec2(1.0,0.0),vec2(t,0.0));
  return unaryLogMantissa(m).x;
}
float unaryTanh(float x) {
  if (abs(x)<0.0001f) return x;
  if (abs(x)>10.0f) return x<0.0?-1.0:1.0;
  // expm1-style cancellation avoidance near zero via an odd Taylor series.
  if (abs(x)<0.0625f) {
    float z=x*x;
    return x+x*z*(-1.0f/3.0f+z*(2.0f/15.0f+z*(-17.0f/315.0f+z*62.0f/2835.0f)));
  }
  float t=unaryExp(-2.0*abs(x));
  float result=(1.0-t)/(1.0+t);
  return x<0.0?-result:result;
}

vec2 unarySoftplusDF(float x) {
  if (x>16.0) return vec2(x,0.0);
  if (x<-17.0) return vec2(unaryExp(x),0.0);
  vec2 t=unaryExpDF(-abs(x));
  vec2 logarithm=unaryLogOnePlusDF(t);
  return unaryDFAdd(vec2(max(x,0.0),0.0),logarithm);
}
float unarySoftplus(float x) { return unarySoftplusDF(x).x; }
float unaryMish(float x) {
  if (isinf(x)&&x<0.0) return unaryNaN();
  if (x>20.0) return x;
  if (x<-17.0) return unaryExpMul(x,x);
  vec2 e=unaryExpDF(-abs(x));
  vec2 twice=unaryDFMul(vec2(2.0,0.0),e);
  vec2 square=unaryDFMul(e,e);
  vec2 numerator,denominator;
  if (x<0.0) {
    numerator=unaryDFAdd(square,twice);
    denominator=unaryDFAdd(numerator,vec2(2.0,0.0));
  } else {
    numerator=unaryDFAdd(vec2(1.0,0.0),twice);
    denominator=unaryDFAdd(numerator,unaryDFMul(vec2(2.0,0.0),square));
  }
  return unaryDFMul(vec2(x,0.0),unaryDFDiv(numerator,denominator)).x;
}
float unaryErf(float x) {
  float t = abs(x);
  if (t < 0.5) {
    float z = x*x;
    float p = 1.0 + z*(-1.0/3.0 + z*(1.0/10.0 + z*(-1.0/42.0 +
        z*(1.0/216.0 + z*(-1.0/1320.0 + z*(1.0/9360.0 - z/75600.0))))));
    return 1.1283791670955126 * x * p;
  }
  if (t > 6.0) return x < 0.0 ? -1.0 : 1.0;
  // Abramowitz and Stegun 7.1.26: absolute approximation error <=1.5e-7.
  float a = 1.0 / (1.0 + 0.3275911*t);
  float p = (((((1.061405429*a - 1.453152027)*a) + 1.421413741)*a
              - 0.284496736)*a + 0.254829592)*a;
  float y = 1.0 - p*exp(-t*t);
  return x < 0.0 ? -y : y;
}

// 256 fractional bits of 2/pi, least-significant limb first. Derived using
// high-precision arithmetic, rather than reducing by an imprecise float pi.
uint unaryTwoOverPi(uint i) {
  if (i==0u) return 0xdebbc561u;
  if (i==1u) return 0xfe5163abu;
  if (i==2u) return 0x3c439041u;
  if (i==3u) return 0xdb629599u;
  if (i==4u) return 0xf534ddc0u;
  if (i==5u) return 0xfc2757d1u;
  if (i==6u) return 0x4e441529u;
  return 0xa2f9836eu;
}
uint unaryProductBits(uint product[9],uint bit) {
  uint word=bit/32u, shift=bit%32u;
  uint value=product[word]>>shift;
  if (shift!=0u && word+1u<9u) value|=product[word+1u]<<(32u-shift);
  return value;
}
struct UnaryAngle { vec2 value; uint quadrant; };
UnaryAngle unaryAngle(vec2 value,uint quadrant) {
  UnaryAngle result;
  result.value=value;
  result.quadrant=quadrant;
  return result;
}
UnaryAngle unaryReducedAngle(float x) {
  if (abs(x)<=0.7853981633974483) return unaryAngle(vec2(x,0.0),0u);
  uint bits=floatBitsToUint(x);
  uint significand=(bits&0x007fffffu)|0x00800000u;
  int exponent=int((bits>>23u)&255u)-127;
  uint product[9];
  uint carry=0u;
  for(uint i=0u;i<8u;++i) {
    uint high,low;
    umulExtended(significand,unaryTwoOverPi(i),high,low);
    uint sum=low+carry;
    if (sum<low) ++high;
    product[i]=sum;
    carry=high;
  }
  product[8]=carry;
  uint binaryPoint=uint(279-exponent);
  uint quadrant=unaryProductBits(product,binaryPoint)&3u;
  uint high=unaryProductBits(product,binaryPoint-32u);
  uint low=unaryProductBits(product,binaryPoint-64u);
  // Four exact uint16 conversions retain 64 fractional bits before the
  // double-float accumulation, instead of discarding the remainder residual.
  vec2 fraction=unaryDFAdd(vec2(float(high>>16u)*1.52587890625e-5f,0.0),
      vec2(float(high&65535u)*2.3283064365386962890625e-10f,0.0));
  fraction=unaryDFAdd(fraction,vec2(float(low>>16u)*3.5527136788005009293556e-15f,0.0));
  fraction=unaryDFAdd(fraction,vec2(float(low&65535u)*5.4210108624275221700373e-20f,0.0));
  if ((high&0x80000000u)!=0u) {
    fraction=unaryDFAdd(fraction,vec2(-1.0,0.0));
    quadrant=(quadrant+1u)&3u;
  }
  vec2 remainder=unaryDFMul(fraction,vec2(1.57079637050628662f,-4.37113882867379289e-08f));
  if ((bits&0x80000000u)!=0u) {
    remainder=vec2(-remainder.x,-remainder.y);
    quadrant=(4u-quadrant)&3u;
  }
  return unaryAngle(remainder,quadrant);
}
struct UnaryTrigonometric { vec2 sine; vec2 cosine; };
UnaryTrigonometric unaryTrig(vec2 sine,vec2 cosine) {
  UnaryTrigonometric result;
  result.sine=sine;
  result.cosine=cosine;
  return result;
}
UnaryTrigonometric unarySinCosDF(float x) {
  UnaryAngle reduction=unaryReducedAngle(x);
  vec2 r=reduction.value,z=unaryDFMul(r,r);
  // Original Taylor coefficients as high/residual pairs. On [-pi/4,pi/4],
  // these degrees leave truncation below 3e-21 and defer rounding to output.
  vec2 ps=vec2(-8.22063507847652135e-18f,-1.68147809685589804e-25f);
  ps=unaryDFAdd(vec2(2.81145735896637036e-15f,-1.04620847397636578e-22f),unaryDFMul(z,ps));
  ps=unaryDFAdd(vec2(-7.6471636098127127e-13f,-1.22007104711782877e-20f),unaryDFMul(z,ps));
  ps=unaryDFAdd(vec2(1.60590443720742826e-10f,-5.35252651156272563e-18f),unaryDFMul(z,ps));
  ps=unaryDFAdd(vec2(-2.50521079436794025e-08f,-4.41762304464836646e-16f),unaryDFMul(z,ps));
  ps=unaryDFAdd(vec2(2.75573188446287531e-06f,3.79357122429722915e-14f),unaryDFMul(z,ps));
  ps=unaryDFAdd(vec2(-0.000198412701138295233f,2.72559687493345582e-12f),unaryDFMul(z,ps));
  ps=unaryDFAdd(vec2(0.00833333376795053482f,-4.34617203337595015e-10f),unaryDFMul(z,ps));
  ps=unaryDFAdd(vec2(-0.166666671633720398f,4.9670538793122887e-09f),unaryDFMul(z,ps));
  vec2 sine=unaryDFAdd(r,unaryDFMul(unaryDFMul(r,z),ps));
  vec2 pc=vec2(-1.56192068145415131e-16f,-1.54044714659419928e-24f);
  pc=unaryDFAdd(vec2(4.77947725613294544e-14f,7.62544404448642983e-22f),unaryDFMul(z,pc));
  pc=unaryDFAdd(vec2(-1.14707453605089604e-11f,-2.37220768923123805e-19f),unaryDFMul(z,pc));
  pc=unaryDFAdd(vec2(2.08767558795841524e-09f,1.10828391474598523e-16f),unaryDFMul(z,pc));
  pc=unaryDFAdd(vec2(-2.75573199814971304e-07f,7.57511220905119492e-15f),unaryDFMul(z,pc));
  pc=unaryDFAdd(vec2(2.48015876422869042e-05f,-3.40699609366681977e-13f),unaryDFMul(z,pc));
  pc=unaryDFAdd(vec2(-0.00138888892251998186f,3.3631094437103215e-11f),unaryDFMul(z,pc));
  pc=unaryDFAdd(vec2(0.0416666679084300995f,-1.24176346982807218e-09f),unaryDFMul(z,pc));
  pc=unaryDFAdd(vec2(-0.5f,0.0f),unaryDFMul(z,pc));
  vec2 cosine=unaryDFAdd(vec2(1.0,0.0),unaryDFMul(z,pc));
  uint quadrant=reduction.quadrant;
  if (quadrant==0u) return unaryTrig(sine,cosine);
  if (quadrant==1u) return unaryTrig(cosine,vec2(-sine.x,-sine.y));
  if (quadrant==2u) return unaryTrig(vec2(-sine.x,-sine.y),vec2(-cosine.x,-cosine.y));
  return unaryTrig(vec2(-cosine.x,-cosine.y),sine);
}
vec2 unarySinCos(float x) {
  UnaryTrigonometric result=unarySinCosDF(x);
  return vec2(result.sine.x,result.cosine.x);
}
float unaryTan(float x) {
  if (isinf(x)) return unaryNaN();
  UnaryTrigonometric result=unarySinCosDF(x);
  return unaryDFDiv(result.sine,result.cosine).x;
}
float unaryAsinSmall(float x) {
  // Original Taylor series, evaluated only for |x|<=1/2.
  // Coefficients are (2n)!/(4^n*(n!)^2*(2n+1)); truncation <2e-12.
  float z=x*x;
  float p=0.00515330968231990458;
  p=0.00574003767084192359+z*p;
  p=0.00644721031188964875+z*p;
  p=0.00731252587359884545+z*p;
  p=0.00839033580961681506+z*p;
  p=0.0097616095291940784+z*p;
  p=0.0115518008961397051+z*p;
  p=0.0139648437500000007+z*p;
  p=0.0173527644230769239+z*p;
  p=0.0223721590909090919+z*p;
  p=0.0303819444444444441+z*p;
  p=0.0446428571428571438+z*p;
  p=0.0749999999999999972+z*p;
  p=0.166666666666666657+z*p;
  return x+x*z*p;
}
float unaryAsin(float x) {
  if (abs(x)>1.0) return unaryNaN();
  if (abs(x)<=0.5) return unaryAsinSmall(x);
  float t=sqrt((1.0-abs(x))*0.5);
  float y=(1.5707962512969970703125-2.0*unaryAsinSmall(t))+7.549789954891882e-8;
  return x<0.0?-y:y;
}
float unaryAcos(float x) {
  if (abs(x)>1.0) return unaryNaN();
  if (x>0.5) return 2.0*unaryAsinSmall(sqrt((1.0-x)*0.5));
  if (x<-0.5) return (3.141592502593994140625-
      2.0*unaryAsinSmall(sqrt((1.0+x)*0.5)))+1.5099579909783764e-7;
  return (1.5707962512969970703125-unaryAsinSmall(x))+7.549789954891882e-8;
}

