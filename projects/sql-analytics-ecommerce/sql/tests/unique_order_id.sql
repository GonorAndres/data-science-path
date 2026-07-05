SELECT COUNT(*) AS violations
FROM (
  SELECT order_id, COUNT(*) AS n
  FROM int_order_revenue
  GROUP BY order_id
  HAVING COUNT(*) > 1
);
