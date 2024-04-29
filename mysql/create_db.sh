# Start the MySQL service
service mysql start

# Wait for the MySQL service to fully start
sleep 10

# MySQL root password (set your desired password here)
ROOT_PASSWORD="root"

# Create the database
mysql -u root -p"${ROOT_PASSWORD}" -e "CREATE DATABASE db_test;"