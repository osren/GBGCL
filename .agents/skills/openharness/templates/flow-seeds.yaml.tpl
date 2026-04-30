flows:
  - id: credit-apply
    name: 授信申请链路
    trigger: 用户发起授信申请
    services:
      - front-service
      - user-center
      - credit-service
      - struct-gateway
    description: 从进件到授信审批及资金编排准备的主链路

  - id: loan-disbursement
    name: 支用放款链路
    trigger: 用户发起支用
    services:
      - front-service
      - credit-service
      - struct-gateway
      - ledger-service
    description: 从支用申请到放款落账的主链路
