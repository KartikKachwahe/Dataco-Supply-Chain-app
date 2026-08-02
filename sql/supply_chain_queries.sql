-- ============================================================
-- DataCo Supply Chain — SQL KPI Queries
-- Used in Deliverable 2: Supply Chain Performance Dashboard
-- Database: SQLite (loaded from cleaned CSV)
-- ============================================================


-- ============================================================
-- KPI 1: On-Time Delivery Rate by Shipping Mode
-- ============================================================
SELECT
    shipping_mode,
    COUNT(*) AS total_orders,
    SUM(CASE WHEN late_delivery_risk = 0 THEN 1 ELSE 0 END) AS on_time_orders,
    ROUND(
        SUM(CASE WHEN late_delivery_risk = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) AS on_time_delivery_rate_pct
FROM orders
GROUP BY shipping_mode
ORDER BY on_time_delivery_rate_pct DESC;


-- ============================================================
-- KPI 2: Average Shipping Delay by Market Region
-- ============================================================
SELECT
    market,
    COUNT(*) AS total_orders,
    ROUND(AVG(shipping_delay_days), 2) AS avg_delay_days,
    ROUND(AVG(days_for_shipping_real), 2) AS avg_actual_shipping_days,
    ROUND(AVG(days_for_shipment_scheduled), 2) AS avg_scheduled_shipping_days
FROM orders
GROUP BY market
ORDER BY avg_delay_days DESC;


-- ============================================================
-- KPI 3: Revenue & Profit by Product Category (Top 15)
-- ============================================================
SELECT
    category_name,
    COUNT(*) AS total_orders,
    ROUND(SUM(sales), 2) AS total_revenue,
    ROUND(SUM(order_profit_per_order), 2) AS total_profit,
    ROUND(AVG(profit_margin_pct), 2) AS avg_profit_margin_pct
FROM orders
GROUP BY category_name
ORDER BY total_revenue DESC
LIMIT 15;


-- ============================================================
-- KPI 4: Customer Segmentation by Order Frequency
-- ============================================================
SELECT
    CASE
        WHEN order_count = 1 THEN 'One-Time'
        WHEN order_count BETWEEN 2 AND 5 THEN 'Occasional'
        WHEN order_count BETWEEN 6 AND 15 THEN 'Regular'
        ELSE 'VIP'
    END AS customer_segment,
    COUNT(*) AS customer_count,
    ROUND(AVG(total_spend), 2) AS avg_lifetime_spend,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value
FROM (
    SELECT
        customer_id,
        COUNT(*) AS order_count,
        SUM(sales) AS total_spend,
        AVG(sales) AS avg_order_value
    FROM orders
    GROUP BY customer_id
) customer_summary
GROUP BY customer_segment
ORDER BY avg_lifetime_spend DESC;


-- ============================================================
-- KPI 5: Monthly Revenue Trend
-- ============================================================
SELECT
    strftime('%Y-%m', order_date_dateorders) AS order_month,
    COUNT(*) AS order_count,
    ROUND(SUM(sales), 2) AS monthly_revenue,
    ROUND(SUM(order_profit_per_order), 2) AS monthly_profit,
    ROUND(SUM(CASE WHEN late_delivery_risk = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS late_delivery_pct
FROM orders
GROUP BY order_month
ORDER BY order_month;


-- ============================================================
-- KPI 6: Fraud Analysis — Suspected Fraud by Category & Region
-- ============================================================
SELECT
    market,
    category_name,
    COUNT(*) AS total_orders,
    SUM(CASE WHEN order_status = 'SUSPECTED_FRAUD' THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(
        SUM(CASE WHEN order_status = 'SUSPECTED_FRAUD' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) AS fraud_rate_pct
FROM orders
GROUP BY market, category_name
HAVING fraud_count > 0
ORDER BY fraud_rate_pct DESC
LIMIT 20;


-- ============================================================
-- KPI 7: Shipping Mode Performance — Cost vs. Speed
-- ============================================================
SELECT
    shipping_mode,
    ROUND(AVG(days_for_shipping_real), 2) AS avg_delivery_days,
    ROUND(AVG(sales), 2) AS avg_order_value,
    ROUND(AVG(order_profit_per_order), 2) AS avg_profit,
    ROUND(AVG(benefit_per_order), 2) AS avg_benefit_per_order,
    COUNT(*) AS total_orders
FROM orders
GROUP BY shipping_mode
ORDER BY avg_delivery_days;
