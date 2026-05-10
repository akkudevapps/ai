namespace ai.Models;   // adjust namespace to match your project

public class NewsArticle
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Summary { get; set; } = string.Empty;
    public string Url { get; set; } = string.Empty;
    public DateTime PublishDate { get; set; } = DateTime.Now;
    public string? ImageUrl { get; set; }
}