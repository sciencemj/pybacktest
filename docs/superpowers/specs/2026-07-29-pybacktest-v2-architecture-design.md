# Pybacktest V2 Architecture and MCP Design

## 문서 상태

- 상태: 대화형 설계 승인 완료
- 작성일: 2026-07-29
- 대상 릴리스: `0.2.0`
- 범위: 핵심 Python 백테스트 라이브러리와 선택적 MCP 어댑터

## 배경

현재 Pybacktest는 작은 프로젝트로 시작한 뒤 데이터 정규화, 신호 계산,
리밸런싱, 체결, 결과, projection과 Streamlit 기능이 점진적으로 추가되었다.
전체 테스트 59개는 통과하지만, 테스트 통과 여부와 별개로 다음 구조적 위험이
확인되었다.

- `Backtest`가 실행 제어, 포트폴리오 상태, 결과 집계, projection과 plotting을
  함께 담당한다.
- `Stock` 생성자가 데이터 다운로드를 수행하며 종목 표현, I/O, 정규화, 기간
  자르기와 plotting을 함께 담당한다.
- 리스트와 문자열 기반 전략 스키마가 실행할 수 없는 조합도 검증 단계에서
  허용한다.
- 전략 생성 단계에서 주문 수량이 현금에 맞춰 미리 축소되어, 같은 시점의
  매도 대금으로 매수해야 하는 경우를 정확하게 처리하지 못한다.
- 기본 실행 경로에 매월 15일 리밸런싱 같은 숨은 동작이 있다.
- 잘못된 주문 종류가 경고 없이 무시되고, 0% 주문이 1주 주문으로 변환될 수
  있다.
- 빈 종목 집합은 명확한 validation error 대신 `IndexError`를 발생시킨다.
- 같은 이름의 전략 결과가 덮어써질 수 있다.
- `run(end_date=...)` 이후의 데이터를 projection이 사용하는 look-ahead가
  재현되었다.
- Streamlit이 구조화된 결과가 아니라 `Backtest`의 내부 가변 상태에 직접
  의존한다.
- 공식 공개 API가 정의되어 있지 않고 core import가 matplotlib와 yfinance를
  함께 불러온다.

V2는 기존 API를 유지하는 내부 리팩터링이 아니다. 기존 `Backtest`, `Stock`,
`StrategyManager`와 JSON 형식을 폐기하고 명시적 객체 모델로 교체하는 clean
break이다.

## 승인된 제품 방향

V2의 제품 방향은 다음과 같다.

1. Python 객체 API를 가장 중요한 사용 방식으로 제공한다.
2. 현재 구현은 백테스트에 집중하지만 데이터, 주문과 브로커 인터페이스는
   향후 실시간 거래 어댑터를 만들 수 있게 설계한다.
3. 일봉과 분봉 OHLCV, 여러 종목, 롱/숏, 수수료, 슬리피지, 시장가와 지정가를
   지원한다.
4. 수백 종목과 수백만 bar observation을 단일 머신에서 처리할 수 있어야
   한다.
5. Streamlit UI 개편은 별도 후속 프로젝트로 분리한다.
6. MCP를 통해 LLM 에이전트가 안전한 전략을 생성하고, 백테스트하고,
   비교·설명할 수 있게 한다.
7. MCP는 임의 Python 코드를 실행하지 않는다. AI는 버전이 있는 구조화
   `StrategySpec`만 생성한다.
8. 강화학습 환경은 이번 범위에서 제외한다. 이후 별도 프로젝트가 공개 엔진
   인터페이스를 이용해 `reset/step/observation/action/reward` 환경을 만든다.
9. 실제 사용 코드와 소스 코드를 읽으면 데이터 시점, 체결 방식, 비용과 위험
   규칙을 알 수 있어야 한다. 숨은 I/O와 숨은 거래 동작은 허용하지 않는다.

## 목표

- 도메인, 실행 엔진, 외부 I/O와 표현 계층을 분리한다.
- 전략이 포트폴리오를 직접 변경하지 못하게 하고 주문 의도만 생성하게 한다.
- 신호와 feature 계산은 벡터화하고 주문·체결·원장은 시간 순서대로 처리한다.
- 체결과 회계 규칙을 명시적이고 결정론적으로 만든다.
- Python 전략과 MCP `StrategySpec`이 동일한 엔진을 사용하게 한다.
- 모든 실행을 재현하고 주문이 발생한 이유를 추적할 수 있게 한다.
- 데이터 소스, 브로커, 위험 정책, 비용 모델과 결과 저장소를 교체 가능하게
  만든다.
- 예상 가능한 거절과 경고는 구조화된 이벤트로 제공한다.
- core 설치에서 yfinance, MCP, Streamlit과 plotting 의존성을 제거한다.

## 비목표

다음 항목은 `0.2.0` 범위에 포함하지 않는다.

- 기존 V1 API 또는 기존 JSON 전략 형식과의 호환 계층
- Streamlit UI 개편
- 실제 브로커와 연결하는 live trading 구현
- 강화학습 환경
- 틱·호가창 기반 거래와 고빈도 거래
- 분산 실행과 클러스터 스케줄링
- 선물, 옵션과 복잡한 파생상품 회계
- 여러 결제 통화를 자동 환산하는 포트폴리오 회계
- MCP에서 임의 Python 코드, 임의 shell 명령 또는 임의 파일 경로 실행
- 원격 MCP transport와 다중 사용자 인증·권한 관리
- 기존 Monte Carlo projection의 core 편입

한 실행은 하나의 base currency를 사용한다. 다른 통화의 자산을 사용할 때는
사용자가 사전에 base currency로 변환한 가격 데이터를 제공해야 한다.
Projection은 이후 공개 `BacktestResult`를 입력으로 사용하는 별도 analytics
모듈로 다시 제공할 수 있다.

## 선택한 아키텍처

V2는 벡터화 데이터 계층과 이벤트 기반 실행 계층을 결합한 하이브리드
아키텍처를 사용한다.

```text
Python Strategy ───────────────┐
StrategySpec → Compiler ───────┼→ BacktestService → Engine → Domain
MCP Client → MCP Adapter ──────┘         │
                                         ├→ MarketData port
                                         ├→ Broker port
                                         ├→ Risk/Cost models
                                         └→ Result/Artifact store
```

지표와 feature는 column 단위로 사전 계산한다. 주문, 체결, 포지션, 현금과
원장은 bar timestamp 순서대로 처리한다. 이를 통해 순수 이벤트 엔진보다
빠르게 대규모 데이터를 처리하면서 순수 벡터 엔진이 표현하기 어려운 지정가,
부분 체결, 자금 제약과 경로 의존 회계를 정확하게 표현한다.

### 고려했지만 선택하지 않은 접근

#### 순수 이벤트 기반 엔진

실시간 거래와 가장 비슷하고 주문 생명주기를 자연스럽게 표현하지만, 모든
시장 값을 Python event 객체로 만드는 비용과 과도한 event class 수가 수백만
bar 목표에 불리하다.

#### 순수 벡터화 엔진

연구 신호와 파라미터 탐색에는 빠르지만 지정가, 부분 체결, 여러 주문의 순서,
현금·margin 제약과 실시간 브로커 호환성을 정확하게 표현하기 어렵다.

## 의존성 규칙

의존성은 항상 바깥쪽에서 안쪽으로 향한다.

```text
adapters → application → engine → domain
                 ↓          ↓
                ports ←─────┘
```

- `domain`은 pandas, yfinance, matplotlib, Streamlit과 MCP를 import하지 않는다.
- `engine`은 구체적인 데이터 공급자나 브로커 구현을 import하지 않는다.
- `application`은 엔진을 조립하고 use case를 제공하지만 UI나 전송 프로토콜을
  알지 못한다.
- `adapters`만 외부 라이브러리와 I/O를 안다.
- MCP 서버는 `BacktestService`만 호출하며 엔진 내부 객체를 직접 변경하지
  않는다.

Python `3.11+`를 지원 대상으로 한다. 타입은 공개 계약이며 strict type
checking 대상이다.

## 제안 패키지 구조

```text
src/pybacktest/
├── __init__.py
├── domain/
│   ├── instruments.py
│   ├── market.py
│   ├── orders.py
│   ├── portfolio.py
│   └── events.py
├── ports/
│   ├── data.py
│   ├── broker.py
│   ├── strategy.py
│   ├── risk.py
│   └── artifacts.py
├── data/
│   ├── dataset.py
│   ├── validation.py
│   ├── features.py
│   └── calendar.py
├── engine/
│   ├── clock.py
│   ├── execution.py
│   ├── accounting.py
│   └── engine.py
├── strategy/
│   ├── base.py
│   ├── intents.py
│   └── components/
├── risk/
│   ├── policies.py
│   └── sizing.py
├── specs/
│   ├── models.py
│   ├── registry.py
│   └── compiler.py
├── application/
│   ├── requests.py
│   ├── service.py
│   └── experiments.py
├── results/
│   ├── models.py
│   ├── metrics.py
│   ├── explain.py
│   └── serialization.py
├── adapters/
│   ├── data/
│   ├── broker/
│   ├── artifacts/
│   └── plotting/
└── mcp/
    ├── server.py
    ├── tools.py
    └── resources.py
```

`pybacktest.__init__`는 소수의 공식 public object만 export한다. 내부 패키지를
가로질러 import해야 사용하는 기능은 공개 API가 아니다.

선택 의존성은 다음과 같이 분리한다.

- `pybacktest[yfinance]`
- `pybacktest[parquet]`
- `pybacktest[plot]`
- `pybacktest[mcp]`

## 도메인 모델

### 값과 식별자

- `InstrumentId`: symbol과 venue를 포함하는 안정적인 식별자
- `Instrument`: quote currency, tick size, lot size와 timezone
- `Money`: 통화와 금액
- `Quantity`: 0 이상의 절대 크기와 단위를 명확히 표현
- `DateRange`: 시작은 포함하고 종료는 제외하는 `[start, end)` 규칙

금액 계산은 통화가 맞지 않으면 실패한다. 수량은 instrument lot size에 맞게
명시적인 rounding policy를 거친다. 매수·매도 방향은 `Quantity`의 부호가
아니라 `OrderSide`로 표현한다. `TargetQuantity`의 목표 포지션만 signed
quantity를 사용한다.

OHLCV와 feature array는 finite `float64`를 사용한다. 체결 이후 장부에
기록되는 `Money`, fee와 position quantity는 `Decimal`로 변환하고 currency
minor unit, tick size와 lot size에 맞춰 quantize한다. 성과 metric은 불변
ledger snapshot에서 `float64`로 계산하며 계산 허용 오차를 metric metadata에
기록한다.

### 시장 데이터

- `Bar`: timestamp, timeframe, open, high, low, close, volume
- `MarketSlice`: 한 timestamp에서 관찰할 수 있는 여러 instrument의 읽기 전용
  view
- `FeatureBuilder`: causal operator만 조합할 수 있는 feature 선언 API
- `FeaturePlan`: 전략이 선언한 feature DAG와 lookback 요구량
- `FeatureSet`: engine이 계산한 timestamp 정렬 feature

도메인 `Bar`는 개별 값의 의미를 정의한다. 대용량 실행은 매 행마다 `Bar`
객체를 새로 만들지 않고, 검증된 columnar `MarketDataSet`의 경량 view를
사용한다. 첫 구현의 canonical in-memory store는 NumPy column array와
timestamp/instrument index이며 Pandas와 Arrow는 adapter에서 변환한다.

### 주문과 체결

- `OrderIntent`: 전략이 요청한 행동
- `Order`: sizing과 risk 검사를 통과한 제출 주문
- `Fill`: 실제 반영되는 체결
- `OrderId`, `FillId`, `RunId`: 실행마다 충돌하지 않는 식별자
- `OrderSide`: `BUY` 또는 `SELL`
- `OrderType`: `MARKET` 또는 `LIMIT`
- `TimeInForce`: `DAY` 또는 `GTC`
- `OrderStatus`: `PENDING`, `ACCEPTED`, `PARTIALLY_FILLED`, `FILLED`,
  `CANCELLED`, `REJECTED`

V2에서 지원하는 intent는 다음과 같다.

- `TargetWeight`
- `TargetQuantity`
- `MarketOrderIntent`
- `LimitOrderIntent`
- `CancelOrderIntent`

Target intent는 `OrderSizer`가 현재 포트폴리오와 가격을 사용해 주문으로
변환한다. 위험 정책은 주문을 승인, 조정 또는 거절할 수 있다. 조정은 원래
수량과 조정 이유를 이벤트에 남기며 조용히 수량을 변경하지 않는다.

### 포트폴리오와 원장

`PortfolioLedger`는 현금, 포지션, 평균 원가, 실현·미실현 손익, 수수료,
노출과 leverage를 변경할 수 있는 유일한 aggregate다. 전략, 브로커와 결과
기록기는 ledger dictionary를 직접 변경하지 못한다.

Ledger는 오직 `Fill`과 명시적인 cash event를 통해 갱신된다. 각 갱신 후 다음
불변식을 검사한다.

- 현금 변화는 체결 금액, 수수료와 외부 cash event로 설명되어야 한다.
- 포지션 변화는 fill 수량의 합과 같아야 한다.
- 허용되지 않은 short, margin과 leverage 상태가 없어야 한다.
- NaN 또는 무한대 금액이 없어야 한다.
- snapshot 시각은 이전 snapshot보다 빠를 수 없다.

`PortfolioSnapshot`은 불변이며 전략과 결과 사용자에게 읽기 전용으로
전달된다.

## Python 전략 계약

Python 사용자는 `Strategy` protocol을 직접 구현한다.

```python
class Strategy(Protocol):
    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        ...

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> Sequence[OrderIntent]:
        ...
```

`build_features()`는 값을 직접 계산하지 않고 이동평균, lag, rolling,
expanding과 같은 causal operator로 lazy plan을 만든다. Engine은 plan을
전체 column에 vectorized 실행한다. 계산된 feature 전체를 저장할 수 있지만
`StrategyContext`에는 현재 timestamp까지의 view만 제공한다. 음수 lag,
centered window와 forward fill처럼 미래 observation을 사용할 수 있는
operation은 feature plan validation에서 거부한다.

Custom `FeatureOperator`는 명시적인 lookback과 output unit을 선언하고 공통
prefix-invariance contract test를 통과해야 한다. Prefix-invariance는 시각
`t`의 결과가 전체 dataset에서 계산했을 때와 `t`까지의 prefix만으로
계산했을 때 동일한지 확인한다. MCP는 allowlist된 operator만 사용한다.
`on_bar()`는 포트폴리오를 변경하지 않고 intent만 반환한다.

전략 인스턴스는 한 run에만 속한다. 여러 전략 또는 여러 파라미터 run은
상태를 공유하지 않는다. `StrategyContext`가 제공하는 기능은 다음으로
제한한다.

- 현재 timestamp
- 현재 및 과거 market view
- 현재 portfolio snapshot
- 활성 주문의 읽기 전용 view
- 구조화된 decision reason 기록

Python 전략 상태는 strategy instance가 소유하며 다른 run과 공유하지 않는다.
컴파일된 전략도 자신의 typed state object를 소유한다. 전략이 미래 index를
요청하면 `LookaheadViolation`을 발생시킨다.

## StrategySpec과 컴포넌트 레지스트리

MCP가 사용하는 `StrategySpec`은 Pydantic discriminated union으로 정의하고
명시적인 `spec_version`을 가진다.

```json
{
  "spec_version": "1",
  "name": "ma-cross",
  "universe": ["XNAS:AAPL", "XNAS:MSFT"],
  "features": {
    "fast": {
      "type": "sma",
      "instrument": "XNAS:AAPL",
      "source": "close",
      "window": 20
    },
    "slow": {
      "type": "sma",
      "instrument": "XNAS:AAPL",
      "source": "close",
      "window": 60
    }
  },
  "rules": [
    {
      "when": {
        "type": "crosses_above",
        "left": {"feature": "fast"},
        "right": {"feature": "slow"}
      },
      "then": {
        "type": "target_weight",
        "instrument": "XNAS:AAPL",
        "weight": 0.5
      }
    }
  ],
  "sizer": {"type": "default"},
  "risk": [
    {"type": "max_position_weight", "value": 0.6}
  ]
}
```

실제 schema는 각 컴포넌트의 필수 필드와 허용 범위를 구체적으로 정의한다.
임의 key, 임의 import path와 표현식 문자열은 허용하지 않는다.

컴파일 흐름은 다음과 같다.

```text
JSON input
→ schema validation
→ semantic validation
→ allowlisted component lookup
→ typed component graph
→ compiled Strategy
```

Semantic validation은 feature reference, window, 단위, universe, 주문 유형,
weight와 risk constraint를 실행 전에 검사한다.

`StrategySpec`의 risk constraint는 전략이 요청하는 추가 제한이다. 서버와
Python 실행자가 설정한 engine-level `RiskPolicy`가 항상 최종 권한을 가지며,
spec은 그 제한을 완화할 수 없다. 두 제한이 함께 있으면 더 엄격한 결정을
적용하고 두 근거를 decision trace에 기록한다.

Python 개발자는 entry point 또는 명시적 registry API로 새 indicator, rule,
sizer와 risk component를 등록할 수 있다. MCP catalog에는 서버 설정에서
allowlist한 컴포넌트만 노출한다. 각 catalog entry는 JSON schema, 설명,
단위, lookback 요구량과 예제를 포함한다.

`StrategySpec` compiler와 사람이 작성한 Python `Strategy`는 같은
`Strategy` protocol, engine과 result model을 사용한다.

## 명시적 공개 API

사용 코드는 실행 가정을 숨기지 않는다.

```python
from pybacktest import (
    BacktestEngine,
    BacktestRequest,
    CalendarPolicy,
    DateRange,
    InstrumentId,
    IntrabarPolicy,
    MetricsConfig,
    Money,
    MovingAverageCross,
    Timeframe,
)
from pybacktest.adapters.broker import (
    NextBarOpenFill,
    NoBorrowCost,
    PerShareCommission,
    SimulatedBroker,
    VolumeShareSlippage,
)
from pybacktest.adapters.data import ParquetDataSource
from pybacktest.risk import LongShortRisk

data = ParquetDataSource("datasets/us_equities")
aapl = InstrumentId(symbol="AAPL", venue="XNAS")
msft = InstrumentId(symbol="MSFT", venue="XNAS")

broker = SimulatedBroker(
    fill_model=NextBarOpenFill(
        intrabar_policy=IntrabarPolicy.CONSERVATIVE,
    ),
    commission=PerShareCommission(0.005),
    slippage=VolumeShareSlippage(max_volume_ratio=0.05),
    borrow_cost=NoBorrowCost(),
)

engine = BacktestEngine(
    data=data,
    broker=broker,
    risk=LongShortRisk(max_leverage=1.5),
)

result = engine.run(
    BacktestRequest(
        strategy=MovingAverageCross(fast=20, slow=60),
        universe=[aapl, msft],
        period=DateRange("2024-01-01", "2025-01-01"),
        timeframe=Timeframe.days(1),
        calendar=CalendarPolicy.union(max_staleness_bars=1),
        metrics=MetricsConfig(
            risk_free_rate=0.0,
            annualization_periods=252,
        ),
        initial_cash=Money.usd(100_000),
        seed=42,
    )
)
```

생성자는 I/O를 수행하지 않는다. `engine.run()`이 데이터를 여는 시점은
명시적으로 문서화하며 외부 네트워크 다운로드는 하지 않는다. 원격 데이터를
사용하려면 별도의 ingestion 단계에서 dataset을 생성한다.

설정은 `dict` 대신 typed configuration object를 사용한다. enum과 단위 있는
value object를 사용하고 잘못된 문자열을 실행 중에 해석하지 않는다. 기본값이
있는 경우 signature와 생성된 API 문서에 드러나야 한다.

소스 코드 품질 규칙은 다음과 같다.

- public class와 protocol은 책임, 시점 규칙, 단위와 불변식을 docstring에
  기록한다.
- 모듈은 하나의 책임만 가진다.
- domain 변경은 domain method를 통해서만 수행한다.
- circular import, monkey patch와 service locator를 사용하지 않는다.
- 예상 가능한 흐름을 예외로 제어하지 않는다.
- 테스트 전용 production class를 만들지 않는다.

## 데이터 입력과 검증

데이터 다운로드와 백테스트 실행을 분리한다.

```text
RemoteDataSource → ingest/normalize → versioned dataset
versioned dataset → BacktestEngine
```

지원 adapter의 공통 contract는 다음을 검증한다.

- 필수 OHLCV column과 numeric dtype
- timezone-aware timestamp
- instrument별 strictly increasing unique timestamp
- `high >= max(open, close)`와 `low <= min(open, close)`
- 음수가 아닌 volume
- 양수인 거래 가격
- 요청한 period와 timeframe
- instrument metadata와 base currency

결측 bar는 자동 생성하지 않는다. `CalendarPolicy`가 `UNION` 또는
`INTERSECTION`을 명시하고, stale price 사용 여부와 최대 허용 기간을
설정한다. 거래 불가능한 instrument는 해당 timestamp에서 주문 체결 대상이
아니다.

Dataset은 content fingerprint, source metadata, timezone, timeframe와
normalization version을 가진다. Engine은 run 중 외부 데이터를 갱신하지
않는다.

## 실행 순서와 시점 규칙

각 run은 다음 순서를 따른다.

1. `BacktestRequest`, `Strategy` 또는 `StrategySpec`, 데이터와 instrument
   metadata를 전체 검증한다.
2. `build_features()`가 만든 causal `FeaturePlan`을 검증하고 vectorized
   계산한다.
3. timestamp 순서로 `MarketSlice`를 만든다.
4. 기존 활성 주문을 현재 bar에서 체결할 수 있는지 broker가 평가한다.
5. fill을 ledger에 반영하고 snapshot을 만든다.
6. 전략이 현재 bar와 최신 snapshot을 관찰해 intent를 생성한다.
7. sizing과 risk policy가 intent를 order 또는 rejection으로 변환한다.
8. 새 order를 broker에 제출한다.
9. recorder가 decision, intent, risk decision, order, fill과 snapshot을
   기록한다.

기본 모델에서 bar 종가를 관찰해 생성한 주문은 동일 bar에 체결되지 않는다.
시장가 주문은 다음 거래 가능한 bar의 시가에 slippage와 fee를 적용해
체결한다.

지정가 주문도 제출 다음 bar부터 활성화된다.

- Buy limit에서 open이 limit 이하이면 open에서 체결한다.
- Buy limit에서 open이 limit보다 높고 low가 limit 이하이면 limit에서
  체결한다.
- Sell limit은 반대 규칙을 사용한다.
- volume participation limit에 따라 부분 체결할 수 있다.
- 같은 timestamp의 주문 우선순위는 제출 timestamp와 `OrderId` 순이다.

OHLCV만으로 intrabar 가격 경로를 알 수 없는 경우 `IntrabarPolicy`를
사용한다. 기본은 전략에 유리한 체결을 선택하지 않는 `CONSERVATIVE`다.
다른 policy를 사용하려면 broker 설정에 명시해야 한다.

`DAY` 주문은 instrument의 session 종료 시 취소되고 `GTC` 주문은 명시적
취소 또는 체결까지 유지된다.

## Broker, 비용과 위험

`Broker` port는 order 제출, 취소와 broker event 조회를 정의한다.
`SimulatedBroker`는 이 contract의 백테스트 구현이다. 향후 live broker
adapter도 같은 order와 event model을 사용하지만 이번 범위에는 포함하지
않는다.

교체 가능한 실행 모델은 다음과 같다.

- `FillModel`
- `CommissionModel`
- `SlippageModel`
- `LiquidityModel`
- `BorrowCostModel`
- `OrderSizer`
- `RiskPolicy`

필수 거래 가정은 `SimulatedBroker` 생성 코드에 드러난다. 수수료나
slippage가 0인 연구가 필요하면 `NoCommission()`과 `NoSlippage()`를
명시적으로 전달한다.

Risk decision은 `APPROVED`, `ADJUSTED`, `REJECTED` 중 하나이며 code, 설명,
원래 수량과 최종 수량을 기록한다. 현금 부족, leverage 한도, short 금지와
최대 포지션 위반은 order rejection 또는 adjustment로 처리한다.

## 결과, 추적성과 설명

`BacktestResult`는 불변 run 결과이며 다음을 제공한다.

- `RunManifest`
- `SummaryMetrics`
- equity curve
- cash와 position history
- order와 fill history
- trades와 realized P&L
- exposure와 turnover
- risk decisions와 order rejections
- typed warnings
- decision trace

기본 summary에는 total return, CAGR, volatility, Sharpe, Sortino, maximum
drawdown, turnover, win rate와 평균 exposure를 포함한다. metric은 계산식,
연율화 기준, risk-free rate와 결측 처리 규칙을 metadata로 제공한다.

각 주문은 다음 causal trace를 가진다.

```text
feature values
→ rule decision
→ intent
→ sizing result
→ risk decision
→ order
→ fill or rejection
→ ledger change
```

Python 사용자는 `result.explain_trade(order_id)`로 구조화된 설명을 얻는다.
설명은 기록된 사실만 사용하며 사후 추측을 생성하지 않는다.

Artifact store는 다음 형식을 사용한다.

- `manifest.json`
- `strategy_spec.json` 또는 Python strategy identity metadata
- `config.json`
- `summary.json`
- `equity.parquet`
- `positions.parquet`
- `orders.parquet`
- `fills.parquet`
- `events.parquet`

## 결정성과 재현성

각 run manifest는 다음을 기록한다.

- library와 schema version
- strategy/spec fingerprint
- dataset fingerprint
- 전체 config
- random seed
- 시작·종료 시각
- adapter와 component identity/version
- artifact checksums

같은 입력과 seed는 같은 order, fill, ledger snapshot과 metric을 생성해야 한다.
난수를 사용하는 모델은 engine이 제공한 RNG만 사용한다. wall-clock time,
dictionary iteration order 또는 외부 네트워크 상태에 결과가 의존하면
실패다.

## 오류와 경고

실행 전에 확인 가능한 문제는 validation 단계에서 한 번에 보고한다.

- `ConfigurationError`
- `StrategySpecError`
- `DataValidationError`
- `ComponentNotAllowedError`
- `UnsupportedInstrumentError`

실행 중 예상되는 시장·거래 결과는 event다.

- `OrderRejected`
- `OrderAdjusted`
- `OrderExpired`
- `PartialFill`
- `DataUnavailable`

회계 불변식 위반, 시간 역행, 미래 데이터 접근과 adapter contract 위반은
즉시 run을 중단한다.

- `AccountingInvariantError`
- `LookaheadViolation`
- `ClockRegressionError`
- `AdapterContractError`

모든 오류, 거절과 경고는 안정적인 machine-readable code와 사람이 읽을 수
있는 message를 함께 제공한다. 문자열 message 파싱을 API로 사용하지 않는다.

## MCP와 AI 에이전트

MCP adapter는 `BacktestService`를 호출하며 다음 도구를 제공한다.

### `list_strategy_components`

허용된 feature, rule, intent, sizer와 risk component catalog를 반환한다.
응답은 설명, JSON schema, 단위, lookback 요구량과 짧은 예제를 포함한다.

### `validate_strategy_spec`

Schema와 semantic validation을 수행하고 오류 위치, code와 수정 가능한
설명을 반환한다. 백테스트는 실행하지 않는다.

### `run_backtest`

검증된 `StrategySpec`, dataset id와 `BacktestRequest`를 받아 run을 만든다.
응답은 bounded summary와 `run_id`를 반환한다. 서버는 임의 filesystem path,
URL 또는 Python 코드를 받지 않는다. 호출자가 앞서
`validate_strategy_spec`을 호출했더라도 `run_backtest`는 동일한 schema와
semantic validation을 다시 수행한다.

### `get_backtest_result`

`run_id`의 상태, summary, artifact link와 warning을 반환한다. 실행 중이면
진행 상태를 반환하며 대형 table을 inline으로 전송하지 않는다.

### `compare_backtests`

여러 `run_id`의 동일 정의 metric, 설정 차이와 strategy lineage를 비교한다.

### `explain_trade`

`run_id`와 `order_id`에 대한 기록된 causal trace를 반환한다.

MCP resource는 다음 URI namespace를 사용한다.

```text
pybacktest://components/catalog
pybacktest://runs/{run_id}/manifest
pybacktest://runs/{run_id}/summary
pybacktest://runs/{run_id}/orders
pybacktest://runs/{run_id}/fills
pybacktest://runs/{run_id}/equity
```

서버는 allowlisted dataset과 component만 사용한다. 요청별 최대 instrument,
bar observation, 동시 run과 artifact 크기를 설정으로 제한한다. Tool 응답은
credential, 개인 식별 정보와 내부 경로를 노출하지 않는다.

`0.2.0`은 로컬 `stdio` MCP transport만 제공한다. 원격 HTTP transport와
사용자별 인증·권한은 별도 보안 설계 없이 추가하지 않는다.

### AI 학습용 실험 기록

각 AI iteration은 다음 구조화 정보를 기록할 수 있다.

- `experiment_id`
- `agent_session_id`
- `parent_run_id`
- 제출한 `StrategySpec`
- validation feedback
- run config와 dataset fingerprint
- summary metric과 rejection count
- 생성된 artifact id
- 다음 iteration과의 lineage

이 기록은 LLM 에이전트가
`전략 생성 → 검증 → 실행 → 비교 → 수정`한 trajectory를 평가하거나 학습
데이터로 변환할 수 있게 한다. 서버는 모델의 숨은 추론을 요구하거나
저장하지 않는다. Prompt나 자연어 feedback 저장은 호출자가 명시적으로
제공하고 데이터 정책이 허용할 때만 수행한다.

## 향후 강화학습 프로젝트 경계

별도 강화학습 프로젝트는 다음 공개 인터페이스만 사용한다.

- 읽기 전용 `MarketSlice`
- `PortfolioSnapshot`
- `OrderIntent`
- `Broker`와 `RiskPolicy`
- `SimulationSession`
- 구조화된 reward input metric

`BacktestEngine.create_session()`은 `SimulationSession`을 반환한다.
`SimulationSession`의 공개 API는 현재 observation 조회, 외부
`OrderIntent`를 받아 한 timestamp를 진행하는 `advance()`, 완료 여부와
최종 결과 생성을 제공한다. `BacktestEngine.run()`도 내부적으로 같은
session API를 사용한다.

강화학습 프로젝트는 이 API를 Gym-style environment로 감싸고 observation,
action encoding과 reward를 정의한다. V2 engine 내부 상태나 MCP tool을
import하지 않으며, core에는 RL framework 의존성을 추가하지 않는다.

## 테스트 전략

### 단위 테스트

- value object와 단위 검사
- data validation
- feature plan과 prefix-invariance
- order state transition
- fill, fee와 slippage 계산
- long/short ledger accounting
- risk decision
- metric 계산
- `StrategySpec` schema와 compiler

### Property 및 불변식 테스트

- fill 합과 position 변화 일치
- 체결 금액·수수료와 cash 변화 일치
- 취소·거절된 주문은 ledger를 변경하지 않음
- order status가 역행하지 않음
- strategy가 미래 feature에 접근하지 못함
- 같은 seed와 입력의 event stream 동일

### Golden scenario

작은 사람이 직접 계산할 수 있는 OHLCV fixture로 market order, limit order,
partial fill, long, short, fee, slippage와 rejection을 검증한다. 예상 order,
fill, cash, position과 metric을 fixture로 고정한다.

### Contract test

모든 `MarketDataSource`, `Broker`, `ArtifactStore`와 MCP tool adapter는 공통
contract suite를 통과해야 한다.

### MCP 보안 테스트

- 임의 Python, import path, URL과 filesystem path 거부
- 등록되지 않은 component 거부
- 요청 크기와 run quota 적용
- bounded response 확인
- artifact 권한과 run id 격리

### 성능 회귀 테스트

Reference benchmark는 단일 머신에서 100개 instrument, 총 1,000,000 bar
observation, 두 개의 rolling feature와 market-order 전략을 사용한다.
데이터 ingestion과 artifact 직렬화를 제외한 engine run 목표는 30초 이하,
peak RSS 1.5GB 이하다. benchmark 기준을 달성한 뒤 CI는 동일 reference
환경에서 median runtime 또는 peak memory가 20% 이상 악화되면 실패한다.
Reference runner의 CPU, memory, OS, Python과 dependency version은 benchmark
결과와 함께 저장한다.

## 완료 조건

`0.2.0` 설계 구현은 다음 조건을 모두 만족할 때 완료된다.

- 기존 API 없이 새 public Python API로 end-to-end 백테스트가 실행된다.
- 일봉과 분봉의 multi-instrument 데이터를 처리한다.
- 롱/숏, 시장가, 지정가, DAY/GTC, 부분 체결, 비용과 slippage를 지원한다.
- 미래 데이터 접근과 같은-bar 종가 주문 체결이 차단된다.
- 모든 ledger 변경이 fill 또는 cash event로 추적된다.
- Python 전략과 `StrategySpec` 전략 결과가 동일 engine semantics를 따른다.
- `run()`과 공개 `SimulationSession`이 같은 event ordering과 ledger
  semantics를 사용한다.
- MCP의 여섯 도구와 resource가 allowlist와 quota를 지킨다.
- run artifact와 causal trade explanation을 재현할 수 있다.
- 단위, property, golden, contract, MCP 보안과 결정성 테스트가 통과한다.
- 1,000,000 observation reference benchmark 목표를 만족한다.
- core import가 yfinance, matplotlib, Streamlit 또는 MCP를 요구하지 않는다.
- README와 API 문서가 명시적 실행 가정을 보여주는 예제를 제공한다.

## 전환 영향

V2는 clean break이므로 기존 README, notebook, tests와 Streamlit app은 새 API를
사용하도록 별도 전환해야 한다. 이번 구현은 core와 MCP까지만 다루므로
Streamlit app은 V2 core 완료 시점에 일시적으로 호환되지 않는다. 기존 UI를
억지로 유지하기 위한 compatibility shim은 만들지 않는다.

기존 V1 코드는 새 domain과 engine이 end-to-end acceptance test를 통과한 뒤
제거한다. 제거 전까지 V1과 V2 구현을 같은 execution path에 혼합하지 않는다.
