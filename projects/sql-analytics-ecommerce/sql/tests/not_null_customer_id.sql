SELECT COUNT(*) AS violations
FROM int_order_revenue
WHERE customer_id IS NULL;
