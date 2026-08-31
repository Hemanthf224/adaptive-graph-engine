# Build Stage
FROM node:18-alpine AS build

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source code and build
COPY . .
RUN npm run build

# Production Stage (Serve via NGINX)
FROM nginx:alpine

# Copy custom NGINX configuration if needed (Optional)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built React app to NGINX's serving directory
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
