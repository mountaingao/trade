SELECT * from stock_rating WHERE rating_date = "2025-04-14"

DELETE  FROM  stock_rating WHERE rating_date = "2025-04-14"



SELECT * from stock_rating_history WHERE rating_date = "2025-04-14"

DELETE  FROM  stock_rating_history WHERE rating_date = "2025-04-14"

SELECT h.symbol,h.stockname,r.total_score,h.total_score FROM stock_rating r,stock_rating_history h WHERE r.symbol = h.symbol AND r.rating_date = h.rating_date
AND r.rating_date =  "2025-04-14"



SELECT * from stock_rating WHERE rating_date = "2025-04-15"

SELECT * from stock_rating WHERE rating_date = "2025-04-18"
AND symbol = "300892"
ORDER BY total_score