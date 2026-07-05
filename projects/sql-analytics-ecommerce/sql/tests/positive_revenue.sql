SELECT COUNT(*) AS violations
FROM int_order_revenue
WHERE order_revenue < 0;
