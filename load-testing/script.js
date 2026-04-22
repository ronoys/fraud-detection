import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "30s", target: 10 },  // ramp up to 10 VUs
    { duration: "1m",  target: 50 },  // hold at 50 VUs
    { duration: "15s", target: 0  },  // ramp down
  ],
  thresholds: {
    http_req_duration: ["p(95)<500"],  // 95th percentile must be under 500ms
    http_req_failed: ["rate<0.01"],    // error rate must be under 1%
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";

function randomFloat(min, max) {
  return Math.random() * (max - min) + min;
}

function generateTransaction() {
  const payload = {
    Time: randomFloat(0, 172792),
    Amount: randomFloat(0, 25000),
  };
  for (let i = 1; i <= 28; i++) {
    payload[`V${i}`] = randomFloat(-3, 3);
  }
  return payload;
}

export default function () {
  const payload = JSON.stringify(generateTransaction());
  const params = { headers: { "Content-Type": "application/json" } };

  const res = http.post(`${BASE_URL}/predict`, payload, params);

  check(res, {
    "status is 200": (r) => r.status === 200,
    "response has fraud field": (r) => {
      try {
        return JSON.parse(r.body).fraud !== undefined;
      } catch {
        return false;
      }
    },
  });

  sleep(0.1);
}
