# Explainable Prediction Report

## Setup
- Model: `dfi-alternatives/results/models/alt3_bipartite_cov.pkl`
- Input dimension: 157
- Test binary vectors available: 352
- Sampled vectors: 10

## Sampled Accuracy
- Correct: 9 / 10
- Accuracy: 0.9000

## Per-sample Explanations

### triplet_1527_right
- True: `right-vs-center` | Pred: `right-vs-center` | Correct: `True`
- P(left-vs-center)=0.1828, P(right-vs-center)=0.8172
- Top EDU-level evidence:
  - f0 val=-1.0 support=+0.2486 :: Center mentions this fact but right side does not: 8l41u2MtPe6yyACw_27 :: Earlier this month , Speaker John Boehner , R-Ohio , announced
  - f1 val=-1.0 support=+0.0499 :: Center mentions this fact but right side does not: 8l41u2MtPe6yyACw_15 :: The Bush-era tax cuts expire along with the payroll tax break ;
  - f8 val=+1.0 support=+0.0380 :: Right-only mention: FwGHYcWhKZFpKuwf_89 :: are the automatic cuts to Pentagon spending .
  - f6 val=+1.0 support=+0.0159 :: Right-only mention: FwGHYcWhKZFpKuwf_77 :: of raising taxes
  - f5 val=-1.0 support=+0.0091 :: Center mentions this fact but right side does not: 8l41u2MtPe6yyACw_37 :: of wanting to raise everyone 's taxes .

### triplet_1631_right
- True: `right-vs-center` | Pred: `right-vs-center` | Correct: `True`
- P(left-vs-center)=0.1519, P(right-vs-center)=0.8481
- Top EDU-level evidence:
  - f0 val=-1.0 support=+0.1517 :: Center mentions this fact but right side does not: itEuDd5E0EubxXuN_17 :: ( $ 15.2 million ) ,
  - f10 val=+1.0 support=+0.0624 :: Right-only mention: TRQA2QLV9kFTwXZQ_10 :: Over the course of the primary and general elections , Clinton 's campaign has hauled in $ 513 million , roughly double what Trump 's has .
  - f2 val=-1.0 support=+0.0415 :: Center mentions this fact but right side does not: itEuDd5E0EubxXuN_19 :: ( $ 14.3 million ) .
  - f1 val=-1.0 support=+0.0380 :: Center mentions this fact but right side does not: itEuDd5E0EubxXuN_13 :: $ 47.5 million .
  - f4 val=-1.0 support=+0.0241 :: Center mentions this fact but right side does not: itEuDd5E0EubxXuN_81 :: who have taken in the largest share of their individual donation totals from these small donors

### triplet_1704_left
- True: `left-vs-center` | Pred: `right-vs-center` | Correct: `False`
- P(left-vs-center)=0.4482, P(right-vs-center)=0.5518
- Top EDU-level evidence:
  - f0 val=-1.0 support=+0.2949 :: Center mentions this fact but left side does not: p0eZ8Zc3wymZ4LhK_43 :: but to cut the deficit .
  - f1 val=-1.0 support=+0.0487 :: Center mentions this fact but left side does not: p0eZ8Zc3wymZ4LhK_32 :: and reduce the deficit in a balanced way . ''
  - f5 val=-1.0 support=+0.0227 :: Center mentions this fact but left side does not: p0eZ8Zc3wymZ4LhK_31 :: to avoid the sequester
  - f6 val=-1.0 support=+0.0052 :: Center mentions this fact but left side does not: p0eZ8Zc3wymZ4LhK_36 :: to avoid the economically harmful consequences of the sequester for a few months . ''
  - f4 val=-1.0 support=+0.0037 :: Center mentions this fact but left side does not: p0eZ8Zc3wymZ4LhK_14 :: that would bring sharp reductions in spending on defense and other programs ,

### triplet_236_left
- True: `left-vs-center` | Pred: `left-vs-center` | Correct: `True`
- P(left-vs-center)=0.8310, P(right-vs-center)=0.1690
- Top EDU-level evidence:
  - f0 val=+1.0 support=+0.3798 :: Left-only mention: MPf3Dpm5Hm7x4OfR_12 :: The hackers had gained access to the state ' s voter database ,
  - f8 val=-1.0 support=+0.0037 :: Center mentions this fact but left side does not: bhmeayIEmmRPLXPq_11 :: targeted by Russian hackers ahead of the election .
  - f10 val=-1.0 support=+0.0018 :: Center mentions this fact but left side does not: bhmeayIEmmRPLXPq_19 :: Homeland Security formally notified election officials in the states
  - f9 val=-1.0 support=+0.0014 :: Center mentions this fact but left side does not: bhmeayIEmmRPLXPq_59 :: Homeland Security is also working with state election officials

### triplet_286_right
- True: `right-vs-center` | Pred: `right-vs-center` | Correct: `True`
- P(left-vs-center)=0.1006, P(right-vs-center)=0.8994
- Top EDU-level evidence:
  - f0 val=-1.0 support=+0.1413 :: Center mentions this fact but right side does not: 8TTHspd76NylUCLN_0 :: House of Representatives investigators are looking into
  - f15 val=+1.0 support=+0.0624 :: Right-only mention: ERTfhQWePSmrJGqy_45 :: The Justice Department appealed that decision ,
  - f2 val=-1.0 support=+0.0433 :: Center mentions this fact but right side does not: 8TTHspd76NylUCLN_26 :: Mr Trump did not testify in the Mueller investigation ,
  - f1 val=-1.0 support=+0.0373 :: Center mentions this fact but right side does not: 8TTHspd76NylUCLN_22 :: lawmakers were also investigating
  - f4 val=-1.0 support=+0.0254 :: Center mentions this fact but right side does not: 8TTHspd76NylUCLN_23 :: whether Mr Trump lied to Mr Mueller during the course of the probe into Russian meddling in the 2016 election .

### triplet_335_right
- True: `right-vs-center` | Pred: `right-vs-center` | Correct: `True`
- P(left-vs-center)=0.2127, P(right-vs-center)=0.7873
- Top EDU-level evidence:
  - f0 val=-1.0 support=+0.1404 :: Center mentions this fact but right side does not: XbmTJTNa1xKzxT9w_121 :: but also to communicate to you a decision of great importance for the life of the Church .
  - f2 val=-1.0 support=+0.0420 :: Center mentions this fact but right side does not: XbmTJTNa1xKzxT9w_159 :: Some History On Papal Resignations .
  - f1 val=-1.0 support=+0.0389 :: Center mentions this fact but right side does not: XbmTJTNa1xKzxT9w_0 :: For the first time in nearly 600 years , a pope is resigning from his post as leader of the Roman Catholic Church .
  - f42 val=+1.0 support=+0.0276 :: Right-only mention: UEt9k7QXjbFUwo97_9 :: to carry out his papal duties .
  - f12 val=-1.0 support=+0.0187 :: Center mentions this fact but right side does not: XbmTJTNa1xKzxT9w_54 :: that in today 's world `` strength of mind and body are necessary ''

### triplet_524_left
- True: `left-vs-center` | Pred: `left-vs-center` | Correct: `True`
- P(left-vs-center)=0.5663, P(right-vs-center)=0.4338
- Top EDU-level evidence:
  - f5 val=+1.0 support=+0.0331 :: Left-only mention: OhY0tNOhCXGKWRVA_8 :: they plan to vote in favor
  - f8 val=+1.0 support=+0.0300 :: Left-only mention: OhY0tNOhCXGKWRVA_78 :: declaring a national emergency
  - f22 val=-1.0 support=+0.0238 :: Center mentions this fact but left side does not: XBDnYN2QW0LOm8wl_1 :: Senate Majority Leader Mitch McConnell has made it clear to President Donald Trump
  - f12 val=+1.0 support=+0.0213 :: Left-only mention: OhY0tNOhCXGKWRVA_81 :: declaring a national emergency
  - f37 val=-1.0 support=+0.0198 :: Center mentions this fact but left side does not: XBDnYN2QW0LOm8wl_187 :: she will support the resolution

### triplet_574_right
- True: `right-vs-center` | Pred: `right-vs-center` | Correct: `True`
- P(left-vs-center)=0.2165, P(right-vs-center)=0.7835
- Top EDU-level evidence:
  - f0 val=-1.0 support=+0.2616 :: Center mentions this fact but right side does not: 5TXN7Muegqo8lYvf_26 :: ( £153bn )
  - f1 val=-1.0 support=+0.0446 :: Center mentions this fact but right side does not: 5TXN7Muegqo8lYvf_16 :: the US is now planning a summit with Chinese Premier Xi Jinping at the US President 's Mar-a-Lago resort in Florida .
  - f10 val=+1.0 support=+0.0310 :: Right-only mention: w2cHDiRjDjmSl6o4_58 :: he is content to dig in with even more tariffs on nearly all Chinese imports .
  - f11 val=+1.0 support=+0.0299 :: Right-only mention: w2cHDiRjDjmSl6o4_52 :: Mr. Trump then slapped 25 % tariffs on more than $ 200 billion worth of Chinese goods .
  - f7 val=+1.0 support=+0.0251 :: Right-only mention: w2cHDiRjDjmSl6o4_29 :: Mr. Trump tweeted .

### triplet_658_left
- True: `left-vs-center` | Pred: `left-vs-center` | Correct: `True`
- P(left-vs-center)=0.6586, P(right-vs-center)=0.3414
- Top EDU-level evidence:
  - f1 val=+1.0 support=+0.1829 :: Left-only mention: gXOJL2999HhQt6Y8_1 :: beating out Republican candidate Rick Saccone in a deeply conservative district
  - f14 val=-1.0 support=+0.0297 :: Center mentions this fact but left side does not: fXcgufrmzz88GRx3_12 :: Lamb told supporters at his election night party shortly before 1 a.m. ,
  - f13 val=-1.0 support=+0.0275 :: Center mentions this fact but left side does not: fXcgufrmzz88GRx3_9 :: than we thought ,
  - f16 val=-1.0 support=+0.0251 :: Center mentions this fact but left side does not: fXcgufrmzz88GRx3_148 :: ice President Pence also traveled to the district .
  - f11 val=-1.0 support=+0.0168 :: Center mentions this fact but left side does not: fXcgufrmzz88GRx3_1 :: The special election House race in Pennsylvania was too close to call Wednesday morning ,

### triplet_89_left
- True: `left-vs-center` | Pred: `left-vs-center` | Correct: `True`
- P(left-vs-center)=0.6479, P(right-vs-center)=0.3521
- Top EDU-level evidence:
  - f4 val=+1.0 support=+0.0645 :: Left-only mention: XaFU9Cm27eVIlkga_9 :: were approved by the committee on party-line votes ,
  - f12 val=-1.0 support=+0.0432 :: Center mentions this fact but left side does not: Hihn4viRSbxOqtqW_39 :: Nadler 's decision led to vocal objection from Republicans on the committee ,
  - f10 val=-1.0 support=+0.0400 :: Center mentions this fact but left side does not: Hihn4viRSbxOqtqW_8 :: In brief remarks after the votes , Nadler said ,
  - f14 val=-1.0 support=+0.0381 :: Center mentions this fact but left side does not: Hihn4viRSbxOqtqW_55 :: If the full House votes to impeach the president ,
  - f6 val=+1.0 support=+0.0331 :: Left-only mention: XaFU9Cm27eVIlkga_42 :: - and a trial in the Senate will follow , this January ,

