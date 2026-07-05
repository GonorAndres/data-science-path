# SQL Drills: 40 Questions for DS Interviews (Mexican market)

SQL is the single highest-yield interview topic in Mexican DS loops. Banks, retail, fintech, and consulting all screen on SQL in the first technical round. Most screens use PostgreSQL or MySQL syntax; BigQuery appears at product companies. These drills cover the question types most commonly asked: window functions, CTEs, ranking, gaps and islands, cohort retention, funnels.

Solutions use PostgreSQL dialect unless noted. BigQuery differences are flagged where they matter.

How to use this file: work through each question with a blank editor, attempt the answer, then expand the solution. Time yourself (5 to 8 minutes per question is the interview pace).

---

## Section 1: Basics (questions 1 to 5)

### 1. Second highest salary

Given `employees(id, name, salary)`, return the second highest salary. If only one salary exists, return `NULL`.

<details><summary>Solution</summary>

```sql
SELECT MAX(salary) AS second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
```

Or with a window:

```sql
SELECT DISTINCT salary
FROM (
  SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS r
  FROM employees
) t
WHERE r = 2;
```

Gotcha: `LIMIT 1 OFFSET 1` fails if the top salary has duplicates.
</details>

### 2. Employees earning more than their manager

Given `employees(id, name, salary, manager_id)`, list employees with salary greater than their manager.

<details><summary>Solution</summary>

```sql
SELECT e.name
FROM employees e
JOIN employees m ON e.manager_id = m.id
WHERE e.salary > m.salary;
```
</details>

### 3. Duplicate emails

Given `users(id, email)`, find emails that appear more than once.

<details><summary>Solution</summary>

```sql
SELECT email
FROM users
GROUP BY email
HAVING COUNT(*) > 1;
```
</details>

### 4. Customers who never ordered

Given `customers(id, name)` and `orders(id, customer_id)`, list customers with no orders.

<details><summary>Solution</summary>

```sql
SELECT c.name
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.id IS NULL;
```

Equivalent with anti-join:

```sql
SELECT name
FROM customers
WHERE id NOT IN (SELECT customer_id FROM orders WHERE customer_id IS NOT NULL);
```

Gotcha: `NOT IN` breaks if the subquery returns a `NULL`. Use `NOT EXISTS` or the `LEFT JOIN / IS NULL` pattern for safety.
</details>

### 5. Department salary expense

Given `employees(id, dept_id, salary)` and `departments(id, name)`, return total salary per department, sorted descending.

<details><summary>Solution</summary>

```sql
SELECT d.name, SUM(e.salary) AS total_salary
FROM departments d
LEFT JOIN employees e ON d.id = e.dept_id
GROUP BY d.name
ORDER BY total_salary DESC NULLS LAST;
```

`LEFT JOIN` keeps departments with no employees. `NULLS LAST` puts them at the end.
</details>

## Section 2: Aggregation edge cases (6 to 10)

### 6. COUNT gotcha

Given `users(id, email)` where `email` can be `NULL`, what do `COUNT(*)`, `COUNT(email)`, and `COUNT(DISTINCT email)` return?

<details><summary>Solution</summary>

- `COUNT(*)` counts rows including `NULL` email.
- `COUNT(email)` counts rows where email is not null.
- `COUNT(DISTINCT email)` counts distinct non-null emails.

Asked in every bank interview. Know this cold.
</details>

### 7. Average with zero handling

Given `sales(date, units, revenue)`, what is the average revenue per unit? Handle division by zero.

<details><summary>Solution</summary>

```sql
SELECT SUM(revenue) * 1.0 / NULLIF(SUM(units), 0) AS avg_rev_per_unit
FROM sales;
```

`NULLIF(x, 0)` returns NULL instead of 0, so the division returns NULL instead of erroring.
</details>

### 8. Percent of total

Given `sales(category, revenue)`, return each category and its percent of total revenue.

<details><summary>Solution</summary>

```sql
SELECT
  category,
  revenue,
  revenue * 100.0 / SUM(revenue) OVER () AS pct_total
FROM sales;
```

Note the empty `OVER ()`: this is a window function with no partition, giving the grand total.
</details>

### 9. Top 3 per group

Given `sales(category, product, revenue)`, return top 3 products by revenue in each category.

<details><summary>Solution</summary>

```sql
SELECT category, product, revenue
FROM (
  SELECT
    category,
    product,
    revenue,
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC) AS rn
  FROM sales
) t
WHERE rn <= 3;
```

Use `DENSE_RANK()` if ties should all qualify; `ROW_NUMBER()` breaks ties arbitrarily.
</details>

### 10. Median

Given `salaries(employee_id, salary)`, return the median salary. No median aggregate function.

<details><summary>Solution</summary>

```sql
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median
FROM salaries;
```

Works in PostgreSQL and BigQuery. For MySQL 8+:

```sql
WITH ranked AS (
  SELECT salary, ROW_NUMBER() OVER (ORDER BY salary) AS rn, COUNT(*) OVER () AS n
  FROM salaries
)
SELECT AVG(salary) AS median
FROM ranked
WHERE rn IN ((n + 1) / 2, (n + 2) / 2);
```
</details>

## Section 3: Window functions (11 to 18)

### 11. Running total

Given `orders(order_date, amount)`, show running total by date.

<details><summary>Solution</summary>

```sql
SELECT
  order_date,
  amount,
  SUM(amount) OVER (ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM orders;
```
</details>

### 12. Week-over-week change

Given `orders(order_date, amount)`, show weekly revenue and week-over-week percent change.

<details><summary>Solution</summary>

```sql
WITH weekly AS (
  SELECT DATE_TRUNC('week', order_date) AS week_start, SUM(amount) AS revenue
  FROM orders
  GROUP BY 1
)
SELECT
  week_start,
  revenue,
  LAG(revenue) OVER (ORDER BY week_start) AS prev_week,
  (revenue - LAG(revenue) OVER (ORDER BY week_start)) * 100.0 / NULLIF(LAG(revenue) OVER (ORDER BY week_start), 0) AS wow_pct
FROM weekly;
```
</details>

### 13. Rolling 7-day average

Given `sales(date, units)`, compute 7-day trailing average.

<details><summary>Solution</summary>

```sql
SELECT
  date,
  units,
  AVG(units) OVER (
    ORDER BY date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS rolling_7d
FROM sales;
```

Gotcha: if dates have gaps, `ROWS` uses row count not calendar days. Use `RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW` for calendar windows (PostgreSQL).
</details>

### 14. First and last value per group

Given `logins(user_id, login_at)`, for each user show first and last login.

<details><summary>Solution</summary>

```sql
SELECT DISTINCT
  user_id,
  FIRST_VALUE(login_at) OVER (PARTITION BY user_id ORDER BY login_at) AS first_login,
  LAST_VALUE(login_at) OVER (
    PARTITION BY user_id
    ORDER BY login_at
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS last_login
FROM logins;
```

Gotcha: `LAST_VALUE` default frame is `UNBOUNDED PRECEDING TO CURRENT ROW`, which gives you the current row, not the last. Explicit frame is required.

Simpler:

```sql
SELECT user_id, MIN(login_at) AS first_login, MAX(login_at) AS last_login
FROM logins
GROUP BY user_id;
```
</details>

### 15. Difference from group average

Given `employees(dept_id, salary)`, show each row with the deviation from its department's average.

<details><summary>Solution</summary>

```sql
SELECT
  dept_id,
  salary,
  AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg,
  salary - AVG(salary) OVER (PARTITION BY dept_id) AS diff_from_avg
FROM employees;
```
</details>

### 16. N-tile buckets

Given `customers(id, ltv)`, bucket customers into 10 deciles by LTV.

<details><summary>Solution</summary>

```sql
SELECT id, ltv, NTILE(10) OVER (ORDER BY ltv) AS decile
FROM customers;
```
</details>

### 17. Sessions from events (30-minute inactivity rule)

Given `events(user_id, event_at)`, split events into sessions where a gap greater than 30 minutes starts a new session.

<details><summary>Solution</summary>

```sql
WITH flagged AS (
  SELECT
    user_id,
    event_at,
    CASE
      WHEN event_at - LAG(event_at) OVER (PARTITION BY user_id ORDER BY event_at) > INTERVAL '30 minutes'
      THEN 1 ELSE 0
    END AS new_session
  FROM events
),
sessioned AS (
  SELECT
    user_id,
    event_at,
    SUM(new_session) OVER (PARTITION BY user_id ORDER BY event_at) AS session_id
  FROM flagged
)
SELECT user_id, session_id, MIN(event_at) AS session_start, MAX(event_at) AS session_end, COUNT(*) AS n_events
FROM sessioned
GROUP BY user_id, session_id;
```

This is the canonical "sessionization" pattern. Expect it at Mercado Libre, Nu, Bitso, Kavak interviews.
</details>

### 18. Rank change between two periods

Given `sales_by_month(product, month, revenue)`, compute rank within month, then show rank change vs previous month.

<details><summary>Solution</summary>

```sql
WITH ranked AS (
  SELECT
    product,
    month,
    revenue,
    RANK() OVER (PARTITION BY month ORDER BY revenue DESC) AS r
  FROM sales_by_month
)
SELECT
  product,
  month,
  r,
  r - LAG(r) OVER (PARTITION BY product ORDER BY month) AS rank_change
FROM ranked;
```
</details>

## Section 4: CTEs, subqueries, recursion (19 to 24)

### 19. Multi-CTE pipeline

Given `orders(id, customer_id, amount, order_date)`, in one query compute: top 10 customers by total spend, their average order amount, and their first order date.

<details><summary>Solution</summary>

```sql
WITH totals AS (
  SELECT customer_id, SUM(amount) AS total_spend
  FROM orders
  GROUP BY customer_id
),
top10 AS (
  SELECT customer_id
  FROM totals
  ORDER BY total_spend DESC
  LIMIT 10
),
enriched AS (
  SELECT
    o.customer_id,
    SUM(o.amount) AS total_spend,
    AVG(o.amount) AS avg_order,
    MIN(o.order_date) AS first_order
  FROM orders o
  JOIN top10 t ON o.customer_id = t.customer_id
  GROUP BY o.customer_id
)
SELECT * FROM enriched
ORDER BY total_spend DESC;
```
</details>

### 20. Recursive org chart

Given `employees(id, name, manager_id)`, list every employee under a given manager (say id 5) recursively.

<details><summary>Solution</summary>

```sql
WITH RECURSIVE reports AS (
  SELECT id, name, manager_id FROM employees WHERE id = 5
  UNION ALL
  SELECT e.id, e.name, e.manager_id
  FROM employees e
  JOIN reports r ON e.manager_id = r.id
)
SELECT * FROM reports;
```
</details>

### 21. NOT EXISTS vs LEFT JOIN

Which is faster for "customers who never ordered" on a large table? Why?

<details><summary>Solution</summary>

Depends on the engine and indexes. In PostgreSQL with an index on `orders.customer_id`, `NOT EXISTS` and `LEFT JOIN / IS NULL` both optimize to anti-join and perform similarly. `NOT IN` is slower and risky with NULLs. Prefer `NOT EXISTS` for readability.
</details>

### 22. Correlated subquery

Given `employees(id, dept_id, salary)`, return each employee's salary as a percent of their department max.

<details><summary>Solution</summary>

Correlated subquery:

```sql
SELECT
  id,
  dept_id,
  salary,
  salary * 100.0 / (SELECT MAX(salary) FROM employees e2 WHERE e2.dept_id = e1.dept_id) AS pct_of_dept_max
FROM employees e1;
```

Cleaner with window:

```sql
SELECT
  id,
  dept_id,
  salary,
  salary * 100.0 / MAX(salary) OVER (PARTITION BY dept_id) AS pct_of_dept_max
FROM employees;
```

Window is faster on most engines. Know both; interviewers sometimes explicitly want the non-window version.
</details>

### 23. First purchase by channel

Given `orders(customer_id, order_date, channel)`, for each customer return the channel of their first purchase.

<details><summary>Solution</summary>

```sql
WITH first_orders AS (
  SELECT
    customer_id,
    channel,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS rn
  FROM orders
)
SELECT customer_id, channel AS first_channel
FROM first_orders
WHERE rn = 1;
```
</details>

### 24. Pivot without PIVOT

Given `sales(month, category, revenue)`, transform to one row per month with columns `books`, `music`, `movies`.

<details><summary>Solution</summary>

```sql
SELECT
  month,
  SUM(CASE WHEN category = 'books' THEN revenue ELSE 0 END) AS books,
  SUM(CASE WHEN category = 'music' THEN revenue ELSE 0 END) AS music,
  SUM(CASE WHEN category = 'movies' THEN revenue ELSE 0 END) AS movies
FROM sales
GROUP BY month;
```

This `SUM(CASE WHEN ...)` pattern is portable and asked in every BI-adjacent interview.
</details>

## Section 5: Cohort, funnel, retention (25 to 32)

### 25. Monthly cohort retention

Given `users(id, signup_date)` and `activity(user_id, activity_date)`, compute the percent of each signup cohort that returned in months 1, 2, 3.

<details><summary>Solution</summary>

```sql
WITH cohort AS (
  SELECT id AS user_id, DATE_TRUNC('month', signup_date) AS cohort_month
  FROM users
),
activity_months AS (
  SELECT DISTINCT user_id, DATE_TRUNC('month', activity_date) AS activity_month
  FROM activity
),
pairs AS (
  SELECT
    c.cohort_month,
    a.activity_month,
    EXTRACT(MONTH FROM AGE(a.activity_month, c.cohort_month))
      + 12 * EXTRACT(YEAR FROM AGE(a.activity_month, c.cohort_month)) AS months_since,
    c.user_id
  FROM cohort c
  JOIN activity_months a USING (user_id)
),
cohort_sizes AS (
  SELECT cohort_month, COUNT(DISTINCT user_id) AS size FROM cohort GROUP BY 1
)
SELECT
  p.cohort_month,
  p.months_since,
  COUNT(DISTINCT p.user_id) AS retained,
  COUNT(DISTINCT p.user_id) * 100.0 / cs.size AS retention_pct
FROM pairs p
JOIN cohort_sizes cs USING (cohort_month)
WHERE p.months_since <= 3
GROUP BY p.cohort_month, p.months_since, cs.size
ORDER BY 1, 2;
```

Core product analytics query. Fintech and e-commerce hire on this.
</details>

### 26. Funnel conversion rates

Given `events(user_id, event_name, event_at)` with events `viewed_product`, `added_to_cart`, `checkout`, `purchased`, compute the conversion rate at each step.

<details><summary>Solution</summary>

```sql
WITH steps AS (
  SELECT
    COUNT(DISTINCT CASE WHEN event_name = 'viewed_product' THEN user_id END) AS viewed,
    COUNT(DISTINCT CASE WHEN event_name = 'added_to_cart' THEN user_id END) AS added,
    COUNT(DISTINCT CASE WHEN event_name = 'checkout' THEN user_id END) AS checkout,
    COUNT(DISTINCT CASE WHEN event_name = 'purchased' THEN user_id END) AS purchased
  FROM events
)
SELECT
  viewed, added, checkout, purchased,
  added * 100.0 / NULLIF(viewed, 0) AS view_to_add,
  checkout * 100.0 / NULLIF(added, 0) AS add_to_checkout,
  purchased * 100.0 / NULLIF(checkout, 0) AS checkout_to_purchase
FROM steps;
```

Caveat: this does not enforce order. For strict funnel (a user must have viewed before adding), use self-joins or `MIN(event_at)` per step per user.
</details>

### 27. Strict-order funnel

Same events, but each step must have happened in order in time.

<details><summary>Solution</summary>

```sql
WITH firsts AS (
  SELECT
    user_id,
    MIN(CASE WHEN event_name = 'viewed_product' THEN event_at END) AS t_view,
    MIN(CASE WHEN event_name = 'added_to_cart' THEN event_at END) AS t_add,
    MIN(CASE WHEN event_name = 'checkout' THEN event_at END) AS t_checkout,
    MIN(CASE WHEN event_name = 'purchased' THEN event_at END) AS t_purchase
  FROM events
  GROUP BY user_id
)
SELECT
  COUNT(*) FILTER (WHERE t_view IS NOT NULL) AS viewed,
  COUNT(*) FILTER (WHERE t_view IS NOT NULL AND t_add > t_view) AS added,
  COUNT(*) FILTER (WHERE t_add IS NOT NULL AND t_checkout > t_add) AS checkout,
  COUNT(*) FILTER (WHERE t_checkout IS NOT NULL AND t_purchase > t_checkout) AS purchased
FROM firsts;
```
</details>

### 28. Active users (DAU, WAU, MAU)

Given `activity(user_id, activity_date)`, compute DAU for each date and the 30-day MAU ratio.

<details><summary>Solution</summary>

```sql
WITH dau AS (
  SELECT activity_date, COUNT(DISTINCT user_id) AS dau
  FROM activity
  GROUP BY activity_date
),
mau AS (
  SELECT
    activity_date,
    COUNT(DISTINCT user_id) OVER (
      ORDER BY activity_date
      RANGE BETWEEN INTERVAL '29 days' PRECEDING AND CURRENT ROW
    ) AS mau
  FROM (SELECT DISTINCT activity_date, user_id FROM activity) a
)
SELECT d.activity_date, d.dau, m.mau, d.dau * 1.0 / NULLIF(m.mau, 0) AS stickiness
FROM dau d JOIN mau m USING (activity_date);
```

Note: `COUNT(DISTINCT ...) OVER (RANGE BETWEEN ...)` is supported in PostgreSQL but not MySQL. In MySQL, use self-join on a 30-day window.
</details>

### 29. Churn in last 60 days

Given `subscriptions(user_id, start_date, end_date)`, find users whose subscription ended in the last 60 days.

<details><summary>Solution</summary>

```sql
SELECT user_id, end_date
FROM subscriptions
WHERE end_date BETWEEN CURRENT_DATE - INTERVAL '60 days' AND CURRENT_DATE;
```
</details>

### 30. N-day retention curve per cohort

Given signup cohorts, for each cohort show the retention rate at days 1, 7, 14, 28.

<details><summary>Solution</summary>

Pattern: join cohort to activity, compute `days_since = activity_date - signup_date`, then `COUNT(DISTINCT user_id)` where `days_since` equals the target interval. Divide by cohort size.

```sql
WITH base AS (
  SELECT u.id AS user_id, u.signup_date, a.activity_date,
         (a.activity_date - u.signup_date) AS days_since
  FROM users u
  JOIN activity a ON a.user_id = u.id
),
cohort_sizes AS (
  SELECT signup_date, COUNT(DISTINCT id) AS cohort_size FROM users GROUP BY 1
)
SELECT
  signup_date,
  SUM(CASE WHEN days_since = 1 THEN 1 ELSE 0 END) * 1.0 / cohort_size AS d1,
  SUM(CASE WHEN days_since = 7 THEN 1 ELSE 0 END) * 1.0 / cohort_size AS d7,
  SUM(CASE WHEN days_since = 14 THEN 1 ELSE 0 END) * 1.0 / cohort_size AS d14,
  SUM(CASE WHEN days_since = 28 THEN 1 ELSE 0 END) * 1.0 / cohort_size AS d28
FROM base
JOIN cohort_sizes USING (signup_date)
GROUP BY signup_date, cohort_size;
```

Caveat: "activity on day 7" has two meanings (exactly day 7, or cumulative through day 7). Clarify with the interviewer before writing.
</details>

### 31. Gaps and islands: consecutive login streaks

Given `logins(user_id, login_date)` with one row per user per login day, compute each user's current streak of consecutive login days.

<details><summary>Solution</summary>

```sql
WITH grouped AS (
  SELECT
    user_id,
    login_date,
    login_date - (ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date))::INTEGER * INTERVAL '1 day' AS grp
  FROM logins
)
SELECT
  user_id,
  MIN(login_date) AS streak_start,
  MAX(login_date) AS streak_end,
  COUNT(*) AS streak_length
FROM grouped
GROUP BY user_id, grp
ORDER BY user_id, streak_end DESC;
```

The trick: subtract row_number (as days) from the date; consecutive dates share the same `grp` value.
</details>

### 32. Latest record per group

Given `price_history(product_id, changed_at, price)`, get the current price for each product.

<details><summary>Solution</summary>

```sql
SELECT DISTINCT ON (product_id)
  product_id, changed_at, price
FROM price_history
ORDER BY product_id, changed_at DESC;
```

`DISTINCT ON` is PostgreSQL-only. Cross-engine version:

```sql
SELECT product_id, changed_at, price
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY changed_at DESC) AS rn
  FROM price_history
) t
WHERE rn = 1;
```
</details>

## Section 6: Date, string, performance (33 to 40)

### 33. Month-over-month growth

Given `sales(order_date, amount)`, compute monthly revenue and MoM growth.

<details><summary>Solution</summary>

```sql
WITH monthly AS (
  SELECT DATE_TRUNC('month', order_date) AS month, SUM(amount) AS revenue
  FROM sales
  GROUP BY 1
)
SELECT
  month,
  revenue,
  LAG(revenue) OVER (ORDER BY month) AS prev_month,
  (revenue - LAG(revenue) OVER (ORDER BY month)) * 100.0 / NULLIF(LAG(revenue) OVER (ORDER BY month), 0) AS mom_pct
FROM monthly;
```
</details>

### 34. Year-over-year same month

Same data, compare this April to last April.

<details><summary>Solution</summary>

```sql
WITH monthly AS (
  SELECT DATE_TRUNC('month', order_date) AS month, SUM(amount) AS revenue
  FROM sales
  GROUP BY 1
)
SELECT
  month,
  revenue,
  LAG(revenue, 12) OVER (ORDER BY month) AS same_month_last_year
FROM monthly;
```
</details>

### 35. Extract weekday and group

Given `orders(order_date)`, count orders per weekday.

<details><summary>Solution</summary>

```sql
SELECT
  TO_CHAR(order_date, 'Day') AS weekday,
  EXTRACT(DOW FROM order_date) AS dow_num,
  COUNT(*) AS n
FROM orders
GROUP BY 1, 2
ORDER BY 2;
```

BigQuery uses `FORMAT_DATE('%A', order_date)` instead.
</details>

### 36. Extract year + month key

Why prefer `DATE_TRUNC('month', date)` over `CONCAT(YEAR(date), MONTH(date))`?

<details><summary>Solution</summary>

`DATE_TRUNC` returns a real date, which sorts chronologically and supports arithmetic. String concatenation gives "20264" (ambiguous) and must be right-padded to sort correctly. Always use `DATE_TRUNC` or `FORMAT_DATE('%Y-%m', ...)` for grouping.
</details>

### 37. String matching for fuzzy email domain

Given `users(email)`, count users per email domain (e.g. `gmail.com`, `ciencias.unam.mx`).

<details><summary>Solution</summary>

```sql
SELECT
  SPLIT_PART(email, '@', 2) AS domain,
  COUNT(*) AS n
FROM users
GROUP BY 1
ORDER BY n DESC;
```

BigQuery: `SPLIT(email, '@')[OFFSET(1)]`.
</details>

### 38. JSON extraction

Given `events(id, payload JSONB)` where payload contains `{"user_id": 42, "action": "click"}`, count clicks per user.

<details><summary>Solution</summary>

PostgreSQL:

```sql
SELECT
  (payload->>'user_id')::INT AS user_id,
  COUNT(*) AS clicks
FROM events
WHERE payload->>'action' = 'click'
GROUP BY 1;
```

BigQuery:

```sql
SELECT
  JSON_VALUE(payload, '$.user_id') AS user_id,
  COUNT(*) AS clicks
FROM events
WHERE JSON_VALUE(payload, '$.action') = 'click'
GROUP BY 1;
```
</details>

### 39. Explain the plan

You have a slow query on a 100M-row table. `SELECT * FROM events WHERE user_id = 42 AND event_name = 'click' ORDER BY event_at DESC LIMIT 100`. No indexes. What do you add?

<details><summary>Solution</summary>

Composite index on `(user_id, event_name, event_at DESC)`. The leading columns filter, and the trailing ordered column lets PostgreSQL use an index-only scan for the `ORDER BY`.

If `event_name` is highly selective on its own, a partial index on `(user_id, event_at DESC) WHERE event_name = 'click'` is even faster but requires the filter to be literal.

Explain the choice in the interview: "I'd EXPLAIN ANALYZE first, confirm it's a sequential scan, then add the composite index matching the WHERE and ORDER BY."
</details>

### 40. Count distinct by chunk (approximate)

You need distinct user count over 1B rows and exact count is too slow. What do you use?

<details><summary>Solution</summary>

PostgreSQL: extension `postgres_fdw` or `hyperloglog`. BigQuery: `APPROX_COUNT_DISTINCT(user_id)`. Redshift: `APPROXIMATE COUNT(DISTINCT user_id)`. Snowflake: `APPROX_COUNT_DISTINCT`.

Tradeoff: roughly 1 to 2 percent error in exchange for 10x to 100x speed. Acceptable for product metrics, not for billing.
</details>

---

## Interview pacing

Typical technical screen: 3 to 5 SQL questions in 45 minutes. Expect one from sections 1 or 2 as a warmup, one window function, one funnel/cohort, one perf or edge case. Show your work: narrate your mental model while writing.

## Additional study resources

- **StrataScratch** (free tier covers most of the above).
- **LeetCode SQL** (50 Easy, 50 Medium; Hard is usually overkill for DS loops).
- **DataLemur** (newer but strong).
- **Mode Analytics SQL tutorial** for cohort and funnel patterns.
- **use-the-index-luke.com** for performance intuition.
